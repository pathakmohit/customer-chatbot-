### **README for GYM-BOT Project**

# GYM-BOT: AI-Powered Chatbot for Gym Assistance 💪🤖

**GYM-BOT** is an AI-driven chatbot designed to provide gym-goers with assistance and interact with users in a natural language format. Built using a combination of NLP and machine learning, GYM-BOT is capable of understanding user queries, responding intelligently, and performing additional tasks like web searches and time retrieval.

---

## **Features**
### **Frontend (GUI)**:
- Built with **Tkinter** to provide a user-friendly graphical interface.
- **Chat History Viewer**: A scrollable text box to display chat history.
- **Input Field**: Type messages and receive instant replies.
- **Functional Buttons**:
  - **Clear Chat**: Clears the current chat history.
  - **Save Chat**: Saves the conversation to a `.txt` file.
  - **Help**: Displays a guide on using the chatbot.
- **Special Commands**:
  - `search <query>`: Opens a web search in the default browser.
  - `time`: Displays the current time.
  - `exit`, `quit`, or `bye`: Ends the conversation.

---

### **Backend (Model Training)**
- **Intent Classification**:
  - Processed intents stored in a JSON file with patterns and responses.
  - Leveraged **NLTK** for text preprocessing (tokenization, lemmatization).
  - Created a **Bag-of-Words** model for representing user inputs.
- **Neural Network Model**:
  - Built using **TensorFlow** with:
    - **Input Layer**: Handles the Bag-of-Words vectors.
    - **Hidden Layers**: Two layers (128 and 64 neurons) with ReLU activation and dropout for regularization.
    - **Output Layer**: Uses softmax activation for multi-class classification.
  - Trained for **400 epochs** using stochastic gradient descent (SGD) for optimization.
- **Data Persistence**:
  - Saved preprocessed data (`words.pkl`, `classes.pkl`) and the trained model (`chatbot_model.h5`) for efficient reuse.

---

## **How It Works**
1. **Training the Model**:
   - Run the `file.py` script to preprocess data and train the model.
   - Outputs a trained model (`chatbot_model.h5`).
2. **Launching the Chatbot**:
   - Run the `gui.py` script to open the GYM-BOT interface.
   - Interact with the chatbot through the GUI.

---

## **Technologies Used**
- **Programming Language**: Python
- **Frontend Framework**: Tkinter
- **Machine Learning**: TensorFlow
- **Natural Language Processing**: NLTK
- **Data Serialization**: JSON, Pickle

---

## **Setup Instructions**
1. Clone this repository:
   ```bash
   git clone https://github.com/pathakmohit/customer-chatbot-.git
   cd customer-chatbot-
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following files are in the project directory:
   - `intents.json`: Contains intent data.
   - `file.py`: For training the chatbot.
   - `gui.py`: For launching the chatbot GUI.
4. Train the model:
   ```bash
   python file.py
   ```
5. Start the chatbot:
   ```bash
   python gui.py
   ```

---

## **Demo**
![GYM-BOT Screenshot](assets/demo_screenshot.png)  
*A sleek, easy-to-use interface with intelligent responses to user queries.*

---

## **Contributing**
Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements.

---

## **License**
This project is open-source and licensed under the MIT License.

---

## **Contact**
**Developer**: Mohit Pathak  
**GitHub**: [https://github.com/pathakmohit](https://github.com/pathakmohit)  
For questions or collaboration opportunities, reach out via GitHub!  

--- 

Enhance your fitness journey with GYM-BOT! 🎉
