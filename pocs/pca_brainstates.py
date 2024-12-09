from gensim.downloader import load
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained word vectors (these are like "brain patterns" for words)
print("🧠 Loading word vectors...")
word_vectors = load("glove-wiki-gigaword-100")  # 100-dimensional vectors

# Let's create some "thought streams" - sequences of related words
thought_streams = {
    "happy": ["joy", "smile", "laugh", "happy", "cheerful", "delight"],
    "sad": ["sorrow", "cry", "tears", "grief", "sadness", "misery"],
    "focused": ["concentrate", "study", "learn", "think", "analyze", "understand"],
}

# Convert words to their vector representations
brain_states = {}
for state, words in thought_streams.items():
    # Get vector for each word that exists in our model
    vectors = [word_vectors[w] for w in words if w in word_vectors]
    brain_states[state] = np.array(vectors)

# Apply PCA to see the "brain state trajectories"
all_vectors = np.vstack(list(brain_states.values()))
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(all_vectors)

# Plot the trajectories
plt.figure(figsize=(12, 8))
current_idx = 0
colors = ["r", "b", "g"]
for (state, vectors), color in zip(brain_states.items(), colors):
    n_words = len(vectors)
    state_reduced = reduced_vectors[current_idx : current_idx + n_words]

    # Plot points
    plt.scatter(
        state_reduced[:, 0], state_reduced[:, 1], label=state, color=color, alpha=0.6
    )

    # Plot trajectory
    for i in range(n_words - 1):
        plt.arrow(
            state_reduced[i, 0],
            state_reduced[i, 1],
            state_reduced[i + 1, 0] - state_reduced[i, 0],
            state_reduced[i + 1, 1] - state_reduced[i, 1],
            color=color,
            alpha=0.3,
            head_width=0.1,
        )

    current_idx += n_words

plt.title("Brain State Trajectories in Thought Space")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.grid(True)
plt.show()

# Let's see how much variance we captured
print(f"\n🧮 Analysis of our thought space:")
print(
    f"First two components explain {pca.explained_variance_ratio_.sum()*100:.1f}% of the variance"
)
