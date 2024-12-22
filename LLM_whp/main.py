from moviepy import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import os
import tempfile
import re

# Отключение предупреждения о символических ссылках
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Загрузка процессора и модели
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.to("cuda" if torch.cuda.is_available() else "cpu")

print("Модель и процессор загружены.")

# Функция для извлечения аудио из видео
def extract_audio(video_path):
    print(f"Извлечение аудио из видео: {video_path}")
    # Загружаем видеофайл
    video = VideoFileClip(video_path)

    # Извлекаем аудио
    audio = video.audio

    # Создаем временный файл для сохранения аудио
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path, codec='libmp3lame')

    # Закрываем видеофайл
    video.close()

    print("Аудио успешно извлечено.")
    return temp_audio_path

# Функция для загрузки аудио из файла
def load_audio(audio_path):
    print(f"Загрузка аудио из файла: {audio_path}")
    speech, sampling_rate = librosa.load(audio_path, sr=16000)
    print("Аудио успешно загружено.")
    return speech

# Функция для разбивки аудио на части
def split_audio(audio, chunk_size=30):
    print(f"Разбивка аудио на части по {chunk_size} секунд.")
    # Разбивка аудио на части по chunk_size секунд
    chunks = [audio[i:i + chunk_size * 16000] for i in range(0, len(audio), chunk_size * 16000)]
    print("Аудио успешно разбито на части.")
    return chunks

# Функция для удаления повторений слов
def remove_repetitions(text):
    words = text.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return ' '.join(unique_words)

# Функция для разделения текста на абзацы
def split_into_paragraphs(text, max_length=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    paragraphs = []
    current_paragraph = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentence]
            current_length = len(sentence)
        else:
            current_paragraph.append(sentence)
            current_length += len(sentence)

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return '\n\n'.join(paragraphs)

def main(video_path, output_file_path):
    print("Начало обработки видео.")

    # Извлечение аудио из видео
    temp_audio_path = extract_audio(video_path)

    # Загрузка аудио из временного файла
    audio = load_audio(temp_audio_path)

    # Удаление временного файла
    os.remove(temp_audio_path)

    # Разбивка аудио на части
    audio_chunks = split_audio(audio, chunk_size=30)  # Разбивка на части по 30 секунд

    # Список для хранения транскрипций
    transcriptions = []

    # Обработка каждой части аудио
    for i, chunk in enumerate(audio_chunks):
        print(f"Обработка части аудио {i + 1}/{len(audio_chunks)}.")
        # Подготовка входных данных
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

        # Установка маски внимания
        inputs["attention_mask"] = torch.ones_like(inputs["input_features"])

        # Генерация текста на русском языке
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

        # Декодирование текста
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)
        print(f"Транскрипция части {i + 1} завершена.")

    # Объединение всех транскрипций
    full_transcription = " ".join(transcriptions)
    print("Объединение всех транскрипций завершено.")

    # Удаление повторений слов
    full_transcription = remove_repetitions(full_transcription)
    print("Повторения слов удалены.")

    # Разделение текста на абзацы
    full_transcription = split_into_paragraphs(full_transcription)
    print("Текст разделен на абзацы.")

    # Запись текста в файл
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(full_transcription)

    print(f"Транскрипция сохранена в файл: {output_file_path}")
    print("Обработка видео завершена.")

if __name__ == "__main__":
    video_path = "tests.mp4"  # Укажите путь к вашему видеофайлу
    output_file_path = "transcription.txt"  # Укажите путь для сохранения текста

    main(video_path, output_file_path)
