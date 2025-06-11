# Hacker News Score Predictor 🚀

This project predicts how many upvotes a Hacker News post will get based on its title. Think of it like teaching a computer to understand what makes a good title!

## How It Works 🧠

1. **Word Understanding**: First, we use a pre-trained model (CBOW) that understands what words mean. It's like having a dictionary that knows how words relate to each other.

2. **Title Processing**: When we get a title, we:
   - Break it into words
   - Look up each word in our dictionary
   - Create a "title vector" by averaging the word meanings
   - This is like creating a fingerprint for the title!

3. **Score Prediction**: We use a simple neural network (like a brain) that:
   - Takes the title vector
   - Goes through a few layers of processing
   - Spits out a predicted score

## Project Structure 📁

```
prediction/
├── models/              # Contains our prediction model
│   └── predictor.py     # The neural network that predicts scores
├── data/               # Data handling
│   └── dataset.py      # Helps load and organize our data
├── utils/              # Helper functions
│   ├── data_processing.py  # Processes titles and creates vectors
│   └── training.py     # Handles model training
└── train.py           # Main script to run everything
```

## How to Run 🏃‍♂️

1. Make sure you're in the `prediction` folder
2. Run:
   ```bash
   python train.py
   ```

The script will:
- Load the pre-trained word meanings
- Process all the titles
- Train the model
- Save the best model it finds

## What's Happening During Training 📈

1. The model looks at lots of titles and their actual scores
2. It tries to guess the score for each title
3. If it's wrong, it learns from its mistake
4. It keeps doing this until it gets pretty good at guessing
5. Every now and then, it shows you some examples of its predictions

## The Model Architecture 🏗️

Our prediction model is like a simple brain with three layers:
1. Input layer (32 neurons) - Takes the title vector
2. Hidden layer (128 neurons) - Processes the information
3. Output layer (1 neuron) - Spits out the predicted score

We also use:
- Dropout (0.2) - Helps prevent overfitting (like not memorizing too much)
- Learning rate scheduling - Makes learning more efficient
- Early stopping - Stops when it's not getting better

## Files Explained 📝

- `train.py`: The main script that runs everything
- `models/predictor.py`: The neural network that makes predictions
- `data/dataset.py`: Helps organize our data for training
- `utils/data_processing.py`: Creates title vectors from words
- `utils/training.py`: Handles the training process

## Output Files 📊

When training completes, you'll get:
- `best_predictor.pth`: The best model we found
- `predictor_info.json`: Information about the training results

## Tips 💡

- The model works best with titles that have words it knows
- It learns from actual upvote scores, so it reflects what people like
- You can adjust the number of epochs in `train.py` to train longer or shorter
- The batch size (32) can be adjusted if you have memory issues

## Requirements 🛠️

- Python 3.x
- PyTorch
- NumPy
- The pre-trained CBOW model (`best_cbow_model.pth`)
- The word-to-index mapping (`word_to_lemma_index.json`)
- The Hacker News data (`hn_data_cleaned.json`)

## What's Next? 🔮

You could try:
1. Adding more layers to the model
2. Using different types of word embeddings
3. Adding more features (like post length, time of day, etc.)
4. Trying different learning rates or batch sizes
5. Adding more evaluation metrics

Remember: The goal is to help understand what makes a good Hacker News title! 🎯 