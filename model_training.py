# model_training.py
import torch
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils import DataCollatorSpeechSeq2SeqWithPadding
import evaluate

def train_model(atc_dataset_train, atc_dataset_valid, feature_extractor, tokenizer):
    model_id = 'openai/whisper-tiny'
    out_dir = 'whisper_tiny_atco2_v2'
    epochs = 10
    batch_size = 16

    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.task = 'transcribe'

    print("Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=0.00001,
        warmup_steps=1000,
        bf16=True,
        fp16=False,
        num_train_epochs=epochs,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=225,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        dataloader_num_workers=2,
        save_total_limit=2,
        lr_scheduler_type='constant',
        seed=42,
        data_seed=42,
        gradient_checkpointing=True,
    )

    print("Initializing data collator...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=feature_extractor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    print("Configuring trainer...")
    metric = evaluate.load('wer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {'wer': wer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=atc_dataset_train,
        eval_dataset=atc_dataset_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving the model...")
    model.save_pretrained(f"{out_dir}/best_model")
    tokenizer.save_pretrained(f"{out_dir}/best_model")
    feature_extractor.save_pretrained(f"{out_dir}/best_model")

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    atc_dataset_train, atc_dataset_valid, feature_extractor, tokenizer = load_and_preprocess_data()
    train_model(atc_dataset_train, atc_dataset_valid, feature_extractor, tokenizer)
