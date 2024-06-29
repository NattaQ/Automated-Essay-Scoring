# Automated-Essay-Scoring
Essay writing is an important method to evaluate student learning and performance. It is also time-consuming for educators to grade by hand. Automated Writing Evaluation (AWE) systems can score essays to supplement an educator’s other efforts. AWEs also allow students to receive regular and timely feedback on their writing. However, due to their costs, many advancements in the field are not widely available to students and educators. Open-source solutions to assess student writing are needed to reach every community with these important educational tools.

# Data set
The competition dataset comprises about 24000 student-written argumentative essays. Each essay was scored on a scale of 1 to 6 (Link to the Holistic Scoring Rubric). Your goal is to predict the score an essay received from its text.

## File and Field Information
 - train.csv - Essays and scores to be used as training data.
    - essay_id - The unique ID of the essay
    - full_text - The full essay response
    - score - Holistic score of the essay on a 1-6 scale
    - Sample of train dataset
      | essay_id | full_text | Score |
      |----------|-----------|-------|
      | 000d118 | Many people have car where they live. The thing they don't know is that when you use a car alot of thing can happenÂ like you can get in accidet orÂ the smoke that the car has is bad to breathÂ on if someone is walk but in VAUBAN,Germany they dont have that proble because 70 percent of vauban's families do not own cars,and 57 percent sold a car to move there. Street parkig ,driveways and home garages are forbiddenÂ on the outskirts of freiburd that near the French and Swiss borders. You probaly won't see a car in Vauban's streets because they are completely "car free" butÂ If some that lives in VAUBAN that owns a car ownership is allowed,but there are only two places that you can park a large garages at the edge of the development,where a car owner buys a space but it not cheap to buy one they sell the space for you car for $40,000 along with a home. The vauban people completed this in 2006 ,they said that this an example of a growing trend in Europe,The untile states and some where else are suburban life from auto use this is called "smart planning". The current efforts to drastically reduce greenhouse gas emissions from tailes the passengee cars are responsible for 12 percent of greenhouse gas emissions in Europe and up to 50 percent in some car intensive in the United States. I honeslty think that good idea that they did that is Vaudan because that makes cities denser and better for walking and in VAUBAN there are 5,500 residents within a rectangular square mile. In the artical David Gold berg said that "All of our development since World war 2 has been centered on the cars,and that will have to change" and i think that was very true what David Gold said because alot thing we need cars to do we can go anyway were with out cars beacuse some people are a very lazy to walk to place thats why they alot of people use car and i think that it was a good idea that that they did that in VAUBAN so people can see how we really don't need car to go to place from place because we can walk from were we need to go or we can ride bycles with out the use of a car. It good that they are doing that if you thik about your help the earth in way and thats a very good thing to. In the United states ,the Environmental protection Agency is promoting what is called "car reduced"communtunties,and the legislators are starting to act,if cautiously. Maany experts expect pubic transport serving suburbs to play a much larger role in a new six years federal transportation bill to approved this year. In previous bill,80 percent of appropriations have by law gone to highways and only 20 percent to other transports. There many good reason why they should do this. | 3 |
- test.csv - The essays to be used as test data. Contains the same fields as train.csv, aside from exclusion of score. (Note: The rerun test set has approximately 8k observations.)
    - Sample of test dataset
      | essay_id | full_text |
      |----------|-----------|
      | 000fe60 | I am a scientist at NASA that is discussing the "face" on mars. I will be explaining how the "face" is a land form. By sharing my information about this isue i will tell you just that. First off, how could it be a martions drawing. There is no plant life on mars as of rite now that we know of, which means so far as we know it is not possible for any type of life. That explains how it could not be made by martians. Also why and how would a martion build a face so big. It just does not make any since that a martian did this. Next, why it is a landform. There are many landforms that are weird here in America, and there is also landforms all around the whole Earth. Many of them look like something we can relate to like a snake a turtle a human... So if there are landforms on earth dont you think landforms are on mars to? Of course! why not? It's just unique that the landform on Mars looks like a human face. Also if there was martians and they were trying to get our attention dont you think we would have saw one by now? Finaly, why you should listen to me. You should listen to me because i am a member of NASA and i've been dealing with all of this stuff that were talking about and people who say martians did this have no relation with NASA and have never worked with anything to relate to this landform. One last thing is that everyone working at NASA says the same thing i say, that the "face" is just a landform. To sum all this up the "face" on mars is a landform but others would like to beleive it's a martian sculpture. Which every one that works at NASA says it's a landform and they are all the ones working on the planet and taking pictures. |

 ## Clean Dataset
 - Download Necessary NLTK Data
   
   <img width="438" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/be1955d0-e58c-42c5-a670-64eb000a0bff">
   
   - nltk.download('punkt'): Downloads the Punkt tokenizer models used for sentence and word tokenization.
   - nltk.download('stopwords'): Downloads the list of common English stopwords.
   - nltk.download('wordnet'): Downloads the WordNet lexical database for English, which is used for lemmatization.
   - nltk.download('averaged_perceptron_tagger'): Downloads the part-of-speech tagger model.
- Load and Clean the Dataset
  
  <img width="839" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/470a407a-77c9-4428-9d6d-793170f9b6b8">

   - pd.read_csv(path): Reads the dataset from a CSV file located at the specified path into a Pandas DataFrame.
   - data.dropna(): Removes any rows with missing values from the DataFrame to ensure the data is complete for processing.
- Define Function to Clean Text

  <img width="467" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/147a793c-b73a-4bc3-be23-4d36f12c22f7">

   - Convert to lowercase: text = text.lower() converts all characters in the text to lowercase to ensure uniformity.
   - Remove punctuation: text = text.translate(str.maketrans('', '', string.punctuation)) removes all punctuation marks from the text.
   - Tokenize text: words = word_tokenize(text) splits the text into individual words (tokens).
   - Remove stopwords: stop_words = set(stopwords.words('english')) creates a set of common English stopwords. words = [word for word in words if word not in stop_words] removes these stopwords from the list of words.
   - Remove numbers: words = [word for word in words if not word.isdigit()] removes any tokens that are purely numbers.
   - Lemmatize words: lemmatizer = WordNetLemmatizer() initializes the lemmatizer. words = [lemmatizer.lemmatize(word) for word in words] reduces each word to its base or root form.
   - Rejoin words: cleaned_text = ' '.join(words) joins the cleaned words back into a single string of text.

 ## Extract features from text

   <img width="865" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/243ced4f-aef2-4be7-9c00-18fad63af6b0">


- Basic Text Features
   - Word Count: features['word_count'] = len(words) calculates the total number of words in the text.
   - Character Count: features['char_count'] = len(text) is commented out. If used, it would calculate the total number of characters in the text.
   - Average Word Length: features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0 calculates the average length of words in the text.
   - Sentence Count: features['sentence_count'] = len(sentences) is commented out. If used, it would calculate the total number of sentences in the text.
   - Average Sentence Length: features['avg_sentence_length'] = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0 calculates the average number of words per sentence.
 
- Lexical Features
   - Unique Words Count: unique_words = set(words) creates a set of unique words.
   - Unique Words Count: features['unique_words_count'] = len(unique_words) calculates the number of unique words in the text.
   - Type-Token Ratio: features['type_token_ratio'] = len(unique_words) / len(words) if words else 0 calculates the ratio of unique words to the total number of words, which is an indicator of lexical diversity.

- Syntactic Features
   - Part-of-Speech Tagging: pos_tags = pos_tag(words) assigns part-of-speech tags to each word in the text.
   - Frequency Distribution of POS Tags: pos_counts = FreqDist(tag for (word, tag) in pos_tags) creates a frequency distribution of the POS tags.
   - Noun Count: features['noun_count'] = pos_counts['NN'] + pos_counts['NNS'] calculates the total count of nouns (both singular and plural).
   - Verb Count: features['verb_count'] = pos_counts['VB'] + pos_counts['VBD'] + pos_counts['VBG'] + pos_counts['VBN'] + pos_counts['VBP'] + pos_counts['VBZ'] calculates the total count of verbs in various tenses.
   - Adjective Count: features['adj_count'] = pos_counts['JJ'] + pos_counts['JJR'] + pos_counts['JJS'] calculates the total count of adjectives (both positive, comparative, and superlative forms).
 
- Readability Features
   - Flesch Reading Ease: features['flesch_reading_ease'] = textstat.flesch_reading_ease(text) calculates the Flesch Reading Ease score, which indicates how easy a text is to read.
   - Flesch-Kincaid Grade Level: features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text) calculates the Flesch-Kincaid Grade Level, which indicates the U.S. school grade level required to understand the text.

## Prepares a dataset for training

- Initialize the RoBERTa Tokenizer

  <img width="391" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/849c0952-07e1-4be3-bedd-99a24b6968c4">

   - RobertaTokenizer.from_pretrained('roberta-base'): Initializes the tokenizer for the roberta-base model from the Hugging Face transformers library. This tokenizer is used to convert text into tokens that can be processed by the RoBERTa model.
- Define the EssayDataset Class
  
  <img width="521" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/27dcc95e-9796-43f8-bcdd-a52332383cd3">

   - Class Initialization:
      - __init__: The constructor initializes the dataset with essays, scores, additional features, a tokenizer, and the maximum token length (max_len).
self.essays, self.scores, self.features, self.tokenizer, self.max_len: These attributes store the provided data and tokenizer.

   - Dataset Length:
      - __len__: Returns the number of essays in the dataset.
        
   - Get Item:
      - __getitem__: Retrieves a single item (essay) from the dataset based on the given index.
      - essay = self.essays[index]: Gets the essay text.
      - score = self.scores[index]: Gets the score for the essay.
      - additional_features = torch.tensor(self.features.iloc[index].values, dtype=torch.float): Retrieves additional features and converts them to a PyTorch tensor.
      - tokens = self.tokenizer.encode(essay, add_special_tokens=True): Tokenizes the essay text and adds special tokens required by RoBERTa.
      - chunks = [tokens[i:i + self.max_len] for i in range(0, len(tokens), self.max_len)]: Splits the tokens into chunks of length max_len.
      - input_ids and attention_masks: Initializes tensors to store input IDs and attention masks for each chunk.
      - For each chunk, it fills input_ids and attention_masks with token IDs and corresponding attention masks.
      - Returns a dictionary containing the essay text, input IDs, attention mask, additional features, and score.
        
- Calculate Class Weights

  <img width="404" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/53db93f1-f452-43ba-931b-9ef7a0693afb">

   - Function Definition: calculate_class_weights(scores) calculates the class weights to handle imbalanced datasets.
   - Class Counts: class_counts = np.bincount(scores.astype(int)) counts the number of occurrences of each class.
   - Total Samples: total_samples = len(scores) gets the total number of samples.
   - Class Weights: class_weights = total_samples / (len(class_counts) * class_counts) calculates the weights for each class by dividing the total number of samples by the product of the number of classes and the count of each class.
   - Return Class Weights: The function returns the calculated class weights.
     
- Prepare the Dataset

  <img width="552" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/68cd384f-bb62-47a2-9c6c-1d605e3d325c">

   - Set Maximum Length: MAX_LEN = 512 defines the maximum token length for RoBERTa.
   - Select Features: features = data[['word_count', 'avg_word_length', 'avg_sentence_length', 'unique_words_count', 'type_token_ratio', 'noun_count', 'verb_count', 'adj_count', 'flesch_reading_ease', 'flesch_kincaid_grade']] selects the additional features to be used.
   - Create Dataset: dataset = EssayDataset(...) creates an instance of the EssayDataset class with the provided essays, scores, features, tokenizer, and maximum token length.

## Split dataset


  <img width="480" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/f486dd0e-e81c-496c-a1eb-41361d970215">

- Split the Data into Training and Validation Sets
   - Function: train_test_split is a utility from sklearn.model_selection used to split the dataset into training and validation sets.
   - Parameters:
      - dataset: The complete dataset that needs to be split.
      - test_size=0.2: Specifies that 20% of the data should be used as the validation set, while the remaining 80% will be used for training.
      - random_state=42: Ensures reproducibility by setting a fixed random seed. This means that every time you run this code with the same dataset, the split will be the same.
   - Output: train_data and val_data are the resulting subsets of the dataset for training and validation, respectively.
     
- Calculate Class Weights Based on the Training Data
   - Extract Training Scores:
      - This line extracts the scores from the training dataset. It iterates through each item in the train_data, retrieves the score, converts it to a Python scalar (using .item()), and stores it in a NumPy array.
   - Calculate Class Weights:
      - This line calculates the class weights using the previously defined calculate_class_weights function. This function takes the training scores as input and returns an array of weights corresponding to each class.
      - Class weights are used to handle class imbalance by assigning a higher penalty to misclassifications of underrepresented classes during training.
   - Convert Class Weights to PyTorch Tensor:
      - This line converts the class weights from a NumPy array to a PyTorch tensor with the data type torch.float. This conversion is necessary because the model and loss functions in PyTorch expect input tensors rather than NumPy arrays.
    
## Set up device for computation

   <img width="515" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/2ac4050b-ab45-4b33-814b-f1925f3cb93f">

-  Define the Model

   - Class Definition: Defines a custom neural network model class RoBERTaRegressor that inherits from nn.Module.

   - __init__ Method: Initializes the model.
      - super(RoBERTaRegressor, self).__init__(): Calls the constructor of the parent class (nn.Module).
      - self.roberta = RobertaModel.from_pretrained(roberta_model_name): Loads a pre-trained RoBERTa model.
      - self.drop = nn.Dropout(p=0.4): Adds a dropout layer with a dropout rate of 0.4 to prevent overfitting.
      - self.fc1 = nn.Linear(self.roberta.config.hidden_size + num_additional_features, 256): Adds a fully connected layer that combines the output of RoBERTa and additional features, producing 256 output features.
      - self.activation1 = nn.ReLU(): Adds a ReLU activation function.
      - self.fc2 = nn.Linear(256, 128): Adds a second fully connected layer with 128 output features.
      - self.activation2 = nn.ReLU(): Adds another ReLU activation function.
      - self.fc3 = nn.Linear(128, 1): Adds a final fully connected layer that outputs a single value (suitable for regression tasks).
        
   - forward Method: Defines the forward pass of the model.
      - batch_size, num_chunks, max_len = input_ids.size(): Gets the batch size, number of chunks, and maximum length from the input IDs.
      - input_ids = input_ids.view(-1, max_len): Reshapes the input IDs to combine the batch and chunk dimensions for processing by RoBERTa.
      - attention_mask = attention_mask.view(-1, max_len): Reshapes the attention mask similarly.
      - outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask): Passes the input IDs and attention mask through the RoBERTa model.
      - pooled_output = outputs[1]: Gets the pooled output (corresponding to the [CLS] token) from the RoBERTa model.
      - pooled_output = pooled_output.view(batch_size, num_chunks, -1): Reshapes the pooled output back to separate the batch and chunk dimensions.
      - pooled_output = pooled_output.mean(dim=1): Averages the pooled outputs across chunks.
      - pooled_output = self.drop(pooled_output): Applies dropout to the pooled output.
      - combined_input = torch.cat((pooled_output, additional_features), dim=1): Concatenates the pooled output with the additional features.
      - output = self.fc1(combined_input): Passes the combined input through the first fully connected layer.
      - output = self.activation1(output): Applies the first ReLU activation function.
      - output = self.fc2(output): Passes the result through the second fully connected layer.
      - output = self.activation2(output): Applies the second ReLU activation function.
      - output = self.fc3(output): Passes the result through the final fully connected layer to get the output value.

-  Initialize the Model
   -  Determine the Number of Additional Features: num_additional_features = features.shape[1] gets the number of additional features by checking the number of columns in the features DataFrame.
   -  Model Initialization: model = RoBERTaRegressor('roberta-base', num_additional_features).to(device) initializes the RoBERTaRegressor model with the specified RoBERTa model and number of additional features, and moves the model to the specified device (GPU or CPU).

## Prepare Early stopping for training stage

   <img width="350" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/74699614-7909-47cc-967b-ac47f0e36847">

   - __init__ Method: Initializes the EarlyStopping class with the following parameters:
      - patience: The number of consecutive epochs with no improvement after which training will be stopped. Default is 5.
      - min_delta: The minimum change in the monitored value (validation loss) to qualify as an improvement. Default is 0.
      - self.best_loss: Tracks the best (lowest) validation loss encountered during training. Initialized to None.
      - self.counter: Counts the number of consecutive epochs with no significant improvement. Initialized to 0.
   - __call__ Method: This method is invoked with the current validation loss val_loss as its argument and determines whether training should be stopped.
      - First Condition: if self.best_loss is None:
         - If self.best_loss is None (which it is initially), it means this is the first epoch. Set self.best_loss to the current val_loss and return False to continue training.
      - Second Condition: elif val_loss < self.best_loss - self.min_delta:
         - If the current val_loss is less than self.best_loss minus self.min_delta, it means there has been a significant improvement in the validation loss.
         - Update self.best_loss to the current val_loss, reset self.counter to 0, and return False to continue training.
   - Else Clause: If there is no significant improvement:
      - Increment self.counter by 1.
      - If self.counter is greater than or equal to self.patience, return True to stop training.
      - Otherwise, return False to continue training.

## Training model
- Create Data Loaders

  <img width="616" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/339d6d83-7289-4978-a9b6-183e41ff02d4">

   - Batch Size: The batch size for training and validation is set to 1.
   - DataLoader:
      - train_loader: Loads the training data with shuffling and specified batch size.
      - val_loader: Loads the validation data with the specified batch size.
      - pin_memory=True: Improves the data transfer speed to the GPU.
      - num_workers=4: Uses 4 subprocesses for data loading.
- Set Up Optimizer, Loss Function, and Utilities
  
  <img width="613" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/3be6886f-dacb-4460-8a73-634e10e112fd">

   - Optimizer: AdamW is used with a learning rate of 2e-5 and weight decay of 1e-5.
   - Loss Function: MSELoss with reduction='none' is used, which means the loss is not averaged or summed over the batch. This allows manual application of sample weights.
   - Scaler: GradScaler from torch.cuda.amp is used for mixed precision training to reduce memory usage.
   - Early Stopping: Initialized with patience=10 and min_delta=0.001.
- Learning Rate Scheduler
  
  <img width="796" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/fa3c73f5-3290-4455-aa4e-e8ea330b95c9">

   - Scheduler: ReduceLROnPlateau reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 2 consecutive epochs. The minimum learning rate is set to 1e-6.
- Initialize Variables
  
  <img width="923" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/054fea1f-caa6-44b4-ac08-c61a8c7e1548">

   - Validation Losses: val_losses stores the validation losses for each epoch.
   - Accumulation Steps: accumulation_steps = 4 is used for gradient accumulation to simulate a larger batch size.
   - Best Validation Loss: best_val_loss initialized to infinity.
   - Best Model Path: best_model_path specifies where the best model will be saved.
   - Class Weights: Converts class_weights to a tensor on the appropriate device.
- Training Loop

  <img width="710" alt="image" src="https://github.com/NattaQ/Automated-Essay-Scoring/assets/115794048/398ad293-66df-4fb2-8bf8-68cf48dbbbac">

   - Epoch Loop: Loops over 101 epochs.
   - Training Mode: Sets the model to training mode with model.train().
   - Initialize Loss: Initializes total_loss to 0 and resets the gradients with optimizer.zero_grad().
   - Batch Loop: Iterates over batches in train_loader.
      - Move to Device: Moves input data to the specified device.
      - Mixed Precision: Uses torch.cuda.amp.autocast() for mixed precision.
      - Model Forward Pass: Computes outputs and loss.
      - Sample Weights: Applies class weights to the loss.
      - Loss Accumulation: Accumulates gradients and normalizes by accumulation_steps.
      - Backward Pass: Scales and backpropagates the gradients.
      - Gradient Accumulation: Updates the optimizer after the specified number of accumulation steps.
      - Total Loss: Accumulates total loss.
   - Average Training Loss: Computes and prints the average training loss.
   - Validation Mode: Sets the model to evaluation mode with model.eval().
   - Validation Loss: Computes validation loss similarly to training loss but without gradient updates.
   - Average Validation Loss: Computes and prints the average validation loss and appends it to val_losses.
   - Save Best Model: Saves the model if the current validation loss is the best observed so far.
   - Learning Rate Scheduler: Updates the learning rate based on validation loss.
   - Early Stopping: Checks for early stopping and breaks the loop if triggered.
     
- Plotting Validation Loss
   - Plotting: Creates a plot of epoch numbers vs. validation loss.
   - Figure Settings: Sets the figure size.
   - Plot: Plots the validation losses with markers and lines.
   - Title and Labels: Adds a title and axis labels.
   - Grid: Adds a grid to the plot.
   - Show: Displays the plot.


