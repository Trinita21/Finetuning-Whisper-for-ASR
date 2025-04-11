# evaluation.py
from transformers import WhisperTokenizer
from datasets import load_dataset
import evaluate

def evaluate_model():
    print("Loading evaluation dataset...")
    atc_dataset_valid = load_dataset('jlvdoorn/atco2-asr-atcosim', split='validation')
    
    print("Loading tokenizer...")
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-tiny', language='English', task='transcribe')
    
    print("Loading evaluation metric (WER)...")
    metric = evaluate.load('wer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {'wer': wer}

    print("Evaluating model...")
    trainer = Seq2SeqTrainer(
        model=model,
        eval_dataset=atc_dataset_valid,
        compute_metrics=compute_metrics
    )
    
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    
    with open('evaluation_results.txt', 'w') as f:
        f.write(str(results))
        print(f"Evaluation results saved to 'evaluation_results.txt'")

if __name__ == "__main__":
    evaluate_model()
