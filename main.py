import numpy as np
import torch
import evaluate
import wandb
import pandas as pd
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    GenerationConfig,
)

MAX_SOURCE_LEN  = 1024
MAX_TARGET_LEN  = 128


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_fn(tokenizer, src_col: str, tgt_col: str):
    def _pp(batch):
        src = batch[src_col]
        tgt = batch[tgt_col]
        model_inputs = tokenizer(src, max_length=MAX_SOURCE_LEN, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt, max_length=MAX_TARGET_LEN, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["id"] = batch["id"]            # <— keep pointer to raw rows
        return model_inputs
    return _pp

# -----------------------------
# Custom Trainer (ROUGE + BLEU)
# -----------------------------
class SumTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleu_metric  = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

    def compute_metrics_full(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 (ignore index) for decoding labels
        preds[preds == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id

        dec_preds  = self.tokenizer.batch_decode(preds,  skip_special_tokens=True)
        dec_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        dec_preds  = [p.strip() for p in dec_preds]
        dec_labels = [l.strip() for l in dec_labels]

        bleu = self.bleu_metric.compute(
            predictions=dec_preds,
            references=[[r] for r in dec_labels]
        )["bleu"]

        rouge = self.rouge_metric.compute(
            predictions=dec_preds,
            references=dec_labels,
            use_stemmer=True
        )

        return {
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "rougeLsum": rouge.get("rougeLsum", rouge["rougeL"]),
        }


# -----------------------------
# Eval callback: also eval on train and log samples
# -----------------------------
class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer: SumTrainer, training_sample_size, eval_sample_size, seed):
        self.trainer = trainer
        self.training_sample_size = training_sample_size
        self.eval_sample_size = eval_sample_size

        self.rng_generator = np.random.default_rng(seed)
        self.running = False

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if not self.running and metrics and any(k.startswith("eval") for k in metrics):
            old_flag = self.trainer.args.predict_with_generate
            old_compute_metrics = self.trainer.compute_metrics

            try:
                self.running = True
                self.trainer.args.predict_with_generate = True
                self.trainer.compute_metrics = self.trainer.compute_metrics_full

                train_dataset_length = len(self.trainer.train_dataset)
                train_indices = self.rng_generator.choice(train_dataset_length, size=min(self.training_sample_size, train_dataset_length), replace=False)
                train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset.select(train_indices.tolist()), metric_key_prefix="train")

                self.trainer.log(train_metrics)

                eval_dataset_length = len(self.trainer.eval_dataset)
                eval_indices = self.rng_generator.choice(eval_dataset_length, size=min(self.eval_sample_size, eval_dataset_length), replace=False)
                eval_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset.select(eval_indices.tolist()))

                self.trainer.log(eval_metrics)
            finally:
                self.running = False
                self.trainer.args.predict_with_generate = old_flag
                self.trainer.compute_metrics = old_compute_metrics

        return control


# -----------------------------
# Main
# -----------------------------
def main():
    wandb_login_key = ""
    model_name = "facebook/bart-large"
    src_col = "article"
    tgt_col = "highlights"

    seed = 42
    set_seed(seed)

    ds = load_dataset("cnn_dailymail", "3.0.0")
    ds = ds.map(lambda ex, idx: {"id": idx}, with_indices=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else None),
        attn_implementation="sdpa"
    )

    model.config.use_cache = True

    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.num_beams = 4
    gen_config.length_penalty = 2.0  # <- lives here now
    gen_config.max_new_tokens = MAX_TARGET_LEN
    gen_config.early_stopping = True

    pp = preprocess_fn(tokenizer, src_col, tgt_col)
    remove_cols = [c for c in ds["train"].column_names if c != "id"]
    tokenized = ds.map(pp, batched=True, remove_columns=remove_cols)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir="checkpoints_bart_cnn",
        eval_strategy="epoch",     # uses validation split
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=True,
        report_to="wandb",
        use_cpu=False,
        generation_config=gen_config,
        dataloader_num_workers=4,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
    )

    if wandb_login_key is not None and wandb.run is None:
        wandb.login(key=wandb_login_key)

        wandb.init(
            project="MZ",
            entity="DP_Gereg",
            config=args.to_dict(),
        )

        args.output_dir = f"{wandb.run.name}-{args.output_dir}"

    trainer = SumTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],  # explicit validation split
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Keep the eval callback (train-split metrics + sample logs)
    trainer.add_callback(TrainEvalCallback(trainer, training_sample_size=256, eval_sample_size=256, seed=seed))

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    print("Training…")
    trainer.train()

    print("Evaluating on test…")

    trainer.args.predict_with_generate = True

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.log(test_metrics)

    # Show a few predictions from test
    pred_output = trainer.predict(tokenized["test"], generation_config=gen_config)
    gen_ids = pred_output.predictions
    label_ids = pred_output.label_ids

    gen_ids[gen_ids == -100] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    test_ids = tokenized["test"]["id"]
    orig_inputs = ds["test"].select(test_ids)[src_col]

    table = wandb.Table(columns=[
        "input_question",
        "prediction",
        "labels",
    ])

    for inp, pred, true in zip(orig_inputs, decoded_preds, decoded_labels):
        table.add_data(inp, pred, true)

    trainer.log({"test_predictions_table": table})

    df = pd.DataFrame(table.data, columns=table.columns)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
