# Text Generation with Recurrent Neural Networks (RNNs) and LSTM ✨

## Project Overview 📚

This project demonstrates the power of **Recurrent Neural Networks (RNNs)** with **Long Short-Term Memory (LSTM)** units to generate text. The goal is to create a model capable of generating coherent and stylistically similar text based on a given corpus. In this implementation, the model is trained on a subset of Shakespeare's text, and the generated sequences resemble the style and structure of the original writings.

By leveraging the trained model, you can generate new text sequences that follow the learned patterns. The length and creativity of the output can be controlled via the temperature parameter, giving you flexibility in the generated content.

---

## Key Concepts and Technologies 🧠💻

- **RNNs (Recurrent Neural Networks)**: RNNs are specialized neural networks designed to handle sequential data. They are perfect for text generation as they can capture dependencies over time.
  
- **LSTM (Long Short-Term Memory)**: LSTMs are a type of RNN designed to overcome long-term dependency issues. They are ideal for modeling text as they can remember past information for longer durations.

- **Text Generation**: The model is trained to predict the next character in a sequence, allowing it to generate text one character at a time, creating meaningful outputs based on learned patterns.

---

## Why This Approach? 🤔

- **LSTM for Long-Term Dependencies**: I chose LSTM networks because they excel at learning long-term dependencies, which is essential for generating coherent text over long sequences.

- **Temperature Control**: The temperature parameter in the text generation function allows you to adjust the "creativity" of the output. Lower values make the model more predictable, while higher values introduce more randomness.

---

## Steps Involved 🛠️

### 1. **Text Preprocessing** 🔄
   - **Loading & Cleaning**: The text dataset (Shakespeare's works) is fetched from an online source and converted to lowercase to simplify the vocabulary. Only a subset of the text (characters between 300,000 and 800,000) is used to speed up training.
   - **Character Encoding**: The text is converted into numerical format using two dictionaries:
     - `char_to_index`: Maps each character to a unique index.
     - `index_to_char`: Maps each index back to its corresponding character.

### 2. **Training Dataset Creation** 🎓
   - **Sequence Generation**: The text is split into sequences of fixed length (e.g., 40 characters) with a step size of 3. Each sequence is paired with the next character as the label for training.
   - This approach helps the model learn character patterns and predict the next character based on the sequence context.

### 3. **Building the Model** 🏗️
   - **LSTM Layer**: The model consists of an LSTM layer with 128 units, which helps capture the sequential patterns in the text.
   - **Dense Output Layer**: A dense layer with a softmax activation function is used to predict the next character in the sequence.
   - **Compilation**: The model is compiled using **categorical crossentropy** loss and **RMSprop** optimizer with a learning rate of 0.01.
   - **Training**: The model is trained on the dataset for 10 epochs, and the learned weights are saved for future use.

### 4. **Text Generation** ✍️
   - **Sampling Function**: A custom sampling function selects the next character based on the model’s output probabilities. The `temperature` parameter controls how creative or deterministic the output will be.
   - **Text Generation Function**: 
     - A random starting point is chosen from the text.
     - The model generates one character at a time, based on the previous sequence, until the desired length is reached.
     - The generated text is returned as the output.

### 5. **Saving and Loading the Model** 💾
   - After training, the model is saved using `model.save()`, and it can be loaded later using `tf.keras.models.load_model()` for generating new text without retraining.

---

## Project Structure 📂

```
PoeticaRNN/
│
├── data/
│   └── dataset.txt               # The text dataset used for training
│
├── model/
│   └── text_genrator.h5          # Saved model after training
│
├── src/
│   └── generate_text.py          # Code for text generation and sampling
│
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## Instructions for Use 🚀

1. **Clone the repository**:
   ```bash
   git clone https://github.com/McWebmo/PoeticaRNN.git
   cd PoeticaRNN
   ```

2. **Install dependencies**:
   Ensure you have Python 3.6+ installed. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**:
   The script will load the pre-trained model and generate text.
   ```bash
   cd src
   python generate_text.py
   ```

4. **Modify parameters**:
   - You can adjust the `length` and `temperature` in the `generate_text` function to control the generated text's length and randomness.

---

## Key Features 🌟

- **Text Generation**: Generate new text that mimics the style of Shakespeare or any other corpus.
- **Customizable Temperature**: Control the creativity of the output with the temperature parameter.
- **Efficient Training**: The model is trained on a subset of the text to allow for quick experimentation and iteration.

---

## Future Improvements 🔮

- **Advanced Models**: Exploring more advanced models like **GPT** or **Transformer** could significantly improve text generation quality and coherence.
- **Larger Datasets**: Using larger and more diverse text datasets will help the model generate a wider variety of text styles and genres.
- **User Interface**: Creating an interactive web interface or CLI tool could make it easier for users to experiment with different text sources and model parameters.

---

## Conclusion 🎯

This project demonstrates the end-to-end process of training a Recurrent Neural Network with LSTM to generate text. From preprocessing and model building to text generation, this project highlights the power of machine learning in creating meaningful, creative outputs. With further improvements and refinements, this model can be adapted to generate text in various genres and styles.