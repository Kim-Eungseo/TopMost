import topmost
from topmost.data import download_dataset

device = "cuda"  # or "cpu"
dataset_dir = "./datasets/Amazon_Review"
# download_dataset('Amazon_Review', cache_path='./datasets')

#%%
dict_dir = './datasets/dict'
# download_dataset('dict', cache_path='./datasets')
dataset = topmost.data.CrosslingualDataset(
    dataset_dir,
    lang1='en',
    lang2='cn',
    dict_path=f'{dict_dir}/ch_en_dict.dat',
    device=device,
    batch_size=128
)
#%%
dataset.pretrained_WE_en.shape, dataset.pretrained_WE_cn.shape

# #%%
# len(dataset.word2id_cn), len(dataset.word2id_en)
#
# #%%
# len(dataset.id2word_cn), len(dataset.id2word_en)

#%%
# import fasttext.util
#
# fasttext.util.download_model('en', if_exists='ignore')
# fasttext.util.download_model('zh', if_exists='ignore')
#
# # Load pretrained FastText models
# # Provide the path to your downloaded models
# en_model_path = 'cc.en.300.bin'  # Path to the English FastText model
# cn_model_path = 'cc.zh.300.bin'  # Path to the Chinese FastText model
#
# # Load models using fasttext
# en_model = fasttext.load_model(en_model_path)
# cn_model = fasttext.load_model(cn_model_path)
#
#
# # Function to get embedding for a word
# def get_embedding(word, language='en'):
#     if language == 'en':
#         return en_model.get_word_vector(word)
#     elif language == 'cn':
#         return cn_model.get_word_vector(word)
#     else:
#         raise ValueError("Unsupported language. Use 'en' for English or 'cn' for Chinese.")


#%%
import numpy as np
from typing import Any


def save_embeddings_to_word2vec_format(
        id2word: Any,  # Replace with your Dictionary class type if available
        embeddings: np.ndarray,
        file_path: str
) -> None:
    # Extract vocabulary size and embedding dimension
    vocab_size, embedding_dim = embeddings.shape

    # Open the file to write embeddings
    with open(file_path, 'w') as f:
        # Write the header line
        f.write(f"{vocab_size} {embedding_dim}\n")

        # Iterate through the dictionary to write each word and its corresponding embedding
        for idx, word in id2word.items():
            # Retrieve the corresponding embedding vector
            embedding_vector = embeddings[idx]
            # Convert the embedding vector to a space-separated string of floating-point numbers
            embedding_str = ' '.join(map(str, embedding_vector))
            # Write the word followed by its embedding
            f.write(f"{word} {embedding_str}\n")


#%%

# Save English embeddings from the dataset
# save_embeddings_to_word2vec_format(dataset.id2word_en, dataset.pretrained_WE_en, 'amazon_en.emb')
# save_embeddings_to_word2vec_format(dataset.id2word_cn, dataset.pretrained_WE_cn, 'amazon_cn.emb')

#%%
# make en_cn dictionary text from word2id and id2word and dataset.trans_matrix_en


#%%
from topmost.models.crosslingual.OxTM.vecmap.map_embeddings import map_supervised_embeddings

map_supervised_embeddings(
    "en-zh.txt",
    'amazon_en.emb',
    'amazon_cn.emb',
    'amazon_en_mapped.emb',
    'amazon_cn_mapped.emb'
)

#%%

import numpy as np


def load_embeddings_from_word2vec_format(file_path: str):
    """
    Load embeddings from a Word2Vec text format file.

    Args:
        file_path (str): File path where the embeddings are saved.

    Returns:
        tuple: A tuple containing:
            - vocabulary (dict): A dictionary mapping words to their indices.
            - embeddings (np.ndarray): The embeddings matrix of shape (vocab_size, embedding_dim).
    """
    # Initialize empty dictionary for vocabulary and a list for embeddings
    vocabulary = {}
    embeddings_list = []

    # Open the file to read embeddings
    with open(file_path, 'r') as f:
        vocab_size, embedding_dim = map(int, f.readline().split())

        # Iterate through each line to extract word and corresponding embedding vector
        for index, line in enumerate(f):
            # Split the line into word and embedding values
            values = line.split()
            word = values[0]
            try:
                embedding_vector = np.array(values[1:], dtype=float)
            except ValueError:
                print(f"Error parsing line {index + 1}: {line}")
                continue

            # Check if the embedding vector has the correct size
            if len(embedding_vector) != embedding_dim:
                print(f"Warning: Skipping word '{word}' due to inconsistent embedding size.")
                continue  # Skip this word if the size is incorrect

            # Add word to vocabulary and embedding vector to the list
            vocabulary[word] = index
            embeddings_list.append(embedding_vector)

    # Convert the list of embeddings into a numpy array
    embeddings = np.array(embeddings_list)

    return vocabulary, embeddings


#%%
en_vocabulary, en_embeddings = load_embeddings_from_word2vec_format('amazon_en_mapped.emb')
de_vocabulary, de_embeddings = load_embeddings_from_word2vec_format('amazon_cn_mapped.emb')

#%%
print("English vocabulary size:", len(en_vocabulary))
print("English embeddings shape:", en_embeddings.shape)
print("German vocabulary size:", len(de_vocabulary))
print("German embeddings shape:", de_embeddings.shape)

#%%

from sklearn.metrics.pairwise import cosine_similarity


def find_top_k_closest_words(word, source_vocab, source_embeddings, target_vocab, target_embeddings, k=10):
    """
    Find the top k the closest words in the target language for a given input word from the source language.

    Args:
        word (str): Input word from the source language.
        source_vocab (dict): Source language vocabulary mapping words to indices.
        source_embeddings (np.ndarray): Source language embeddings matrix.
        target_vocab (dict): Target language vocabulary mapping words to indices.
        target_embeddings (np.ndarray): Target language embeddings matrix.
        k (int): Number of closest words to retrieve.

    Returns:
        list: Top k the closest words in the target language.
    """
    # Check if the word exists in the source vocabulary
    if word not in source_vocab:
        print(f"Word '{word}' not found in the source vocabulary.")
        return []

    # Retrieve the embedding of the input word
    source_index = source_vocab[word]
    source_embedding = source_embeddings[source_index].reshape(1, -1)  # Reshape to 2D for cosine similarity

    # Compute cosine similarity between the source embedding and all target embeddings
    similarities = cosine_similarity(source_embedding, target_embeddings).flatten()

    # Find the indices of the top k most similar embeddings
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Retrieve the corresponding words from the target vocabulary
    target_words = list(target_vocab.keys())
    top_k_words = [target_words[idx] for idx in top_k_indices]

    return top_k_words


#%%

def get_closest_words(input_word):
    # Find the top 10 closest words in the German vocabulary for the input English word
    closest_german_words = find_top_k_closest_words(
        word=input_word,
        source_vocab=en_vocabulary,
        source_embeddings=en_embeddings,
        target_vocab=de_vocabulary,
        target_embeddings=de_embeddings,
        k=10
    )

    print(f"Top 10 closest words in Chinese for '{input_word}': {closest_german_words}")


#%%
# reset the pretrained_WE_en and pretrained_WE_cn to the original values with id2word and word2id
for i in range(len(dataset.id2word_en)):
    dataset.pretrained_WE_en[i] = en_embeddings[i]

for i in range(len(dataset.id2word_cn)):
    dataset.pretrained_WE_cn[i] = de_embeddings[i]


#%%
# create a model
model = topmost.models.OxTM(
    num_topics=50,
    vocab_size_en=dataset.pretrained_WE_en.shape[0],
    vocab_size_cn=dataset.pretrained_WE_cn.shape[0],
    pretrain_word_embeddings_en=dataset.pretrained_WE_en,
    pretrain_word_embeddings_cn=dataset.pretrained_WE_cn,
    en1_units=200,
    device_BWE=device
)
model = model.to(device)
#
# # create a trainer
# trainer = topmost.trainers.CrosslingualTrainer(model, dataset, lr_scheduler='StepLR', lr_step_size=125, epochs=500)
# # train the model
# top_words_en, top_words_cn, train_theta_en, train_theta_cn = trainer.train()


# # create a model
# model = topmost.models.InfoCTM(
#     trans_e2c=dataset.trans_matrix_en,
#     pretrain_word_embeddings_en=dataset.pretrained_WE_en,
#     pretrain_word_embeddings_cn=dataset.pretrained_WE_cn,
#     vocab_size_en=dataset.vocab_size_en,
#     vocab_size_cn=dataset.vocab_size_cn,
#     weight_MI=50
# )
# model = model.to(device)

# create a trainer
trainer = topmost.trainers.CrosslingualTrainer(model, dataset, lr_scheduler='StepLR', lr_step_size=125, epochs=500)
# train the model
top_words_en, top_words_cn, train_theta_en, train_theta_cn = trainer.train()

import json


#%%
########################### Evaluate ####################################
from topmost import evaluations

# get theta (doc-topic distributions)
train_theta_en, train_theta_cn, test_theta_en, test_theta_cn = trainer.export_theta()

# compute topic coherence (CNPMI)
# refer to https://github.com/BobXWu/CNPMI

# compute topic diversity
TD = evaluations.multiaspect_topic_diversity((top_words_en, top_words_cn))
print(f"TD: {TD:.5f}")

# evaluate classification
results = evaluations.crosslingual_classification(
    train_theta_en,
    train_theta_cn,
    test_theta_en,
    test_theta_cn,
    dataset.train_labels_en,
    dataset.train_labels_cn,
    dataset.test_labels_en,
    dataset.test_labels_cn,
    classifier="SVM",
    gamma="auto"
)

print(results)

print(top_words_cn)
print(top_words_en)

with open("data.json", "w") as w:
    json.dump([top_words_en, top_words_cn], w, ensure_ascii=False)


