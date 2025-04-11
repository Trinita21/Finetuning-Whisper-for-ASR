# data_preprocessing.py
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm

def load_and_preprocess_data():
    print("Loading training and validation datasets...")
    atc_dataset_train = load_dataset('jlvdoorn/atco2-asr-atcosim', split='train')
    atc_dataset_valid = load_dataset('jlvdoorn/atco2-asr-atcosim', split='validation')

    print(f"Training dataset: {len(atc_dataset_train)} samples")
    print(f"Validation dataset: {len(atc_dataset_valid)} samples")

    print("Casting audio columns to correct sampling rate...")
    atc_dataset_train = atc_dataset_train.cast_column('audio', Audio(sampling_rate=16000))
    atc_dataset_valid = atc_dataset_valid.cast_column('audio', Audio(sampling_rate=16000))

    print("Loading feature extractor and tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-tiny')
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-tiny', language='English', task='transcribe')

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
        batch['labels'] = tokenizer(batch['text']).input_ids
        return batch

    print("Processing training data...")
    atc_dataset_train = atc_dataset_train.map(prepare_dataset, num_proc=1)

    print("Processing validation data...")
    atc_dataset_valid = atc_dataset_valid.map(prepare_dataset, num_proc=1)

    print("Data preprocessing completed!")
    return atc_dataset_train, atc_dataset_valid, feature_extractor, tokenizer

if __name__ == "__main__":
    load_and_preprocess_data()
