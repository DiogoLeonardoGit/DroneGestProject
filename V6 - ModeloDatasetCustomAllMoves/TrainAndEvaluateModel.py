import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from itertools import cycle
import io

# Step 1: Read the Data Files
filename = input("Enter the dataset filename: ")
data = pd.read_csv('dataset/' + filename + '.csv')
validation = pd.read_csv('dataset/validation_' + filename + '.csv')

le = preprocessing.LabelEncoder()


# Function to reshape the data
def reshape_data(data):
    labels = validation['Gesture_id']
    data = data.drop(['Group_id', 'Time(ms)'], axis=1)
    data = data.values.reshape(-1, 64, 6)
    le.fit(labels)
    labels_encoded = le.transform(labels)
    return data, labels_encoded


data, labels_encoded = reshape_data(data)
print("Shape of data:", data.shape)
print("Shape of labels:", labels_encoded.shape)

shape = data.shape
unique_gestures = len(np.unique(labels_encoded))
print("Number of unique gestures: ", unique_gestures)


def plot_gesture_count():
    label_counts = pd.Series(labels_encoded).value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Gestures Count')
    plt.xlabel('Gesture')
    plt.ylabel('Count')
    plt.savefig("gesture_count.png")
    plt.show()
    plt.close()


plot_gesture_count()

x_train, x_val, y_train, y_val = train_test_split(data, labels_encoded, test_size=0.2, random_state=0)


def evaluate_model(model, x_val, y_val):
    y_pred_probabilities = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    accuracy = accuracy_score(y_val, y_pred_classes)
    fscore = f1_score(y_val, y_pred_classes, average='weighted')
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(unique_gestures):
        fpr[i], tpr[i], _ = roc_curve(y_val, y_pred_probabilities[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    class_report = classification_report(y_val, y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    return accuracy, fscore, fpr, tpr, roc_auc, class_report, conf_matrix


def build_model_name(prefix, EPOCH, BATCH, LEARNING_RATE, accuracy):
    model_name = prefix + "_" + EPOCH + "_" + BATCH + "_" + LEARNING_RATE + "_" + accuracy[:5] + ".keras"
    return model_name

def generate_report(model_name, accuracy, fscore, fpr, tpr, roc_auc, history, class_report, conf_matrix):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    c.setFont("Helvetica", 12)

    def add_page():
        c.showPage()
        c.setFont("Helvetica", 12)

    def draw_text(x, y, text):
        c.drawString(x, y, text)
        return y - 15

    def draw_image(image_path, x, y, width, height):
        c.drawImage(image_path, x, y, width, height)
        return y - height - 10

    # Title and Summary
    y = draw_text(margin, height - margin, f"Model Report: {model_name}")
    y = draw_text(margin, y, f"Number of Classes: {unique_gestures}")
    y = draw_text(margin, y, f"Accuracy: {accuracy:.4f}")
    y = draw_text(margin, y, f"F-Score: {fscore:.4f}")

    # add the gesture count plot file to the report
    y = draw_text(margin, y - 10, "Gesture Count Plot:")
    y = draw_image("gesture_count.png", margin, y - 350, width - 2 * margin, 350)

    # add second page
    add_page()
    y = height - margin

    # Classification Report
    y = draw_text(margin, y - 10, "Classification Report:")
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            y = draw_text(margin + 20, y, f"\nClass: {label}")
            for metric, value in metrics.items():
                y = draw_text(margin + 40, y, f"{metric}: {value:.4f}")
        if y < 60:
            add_page()
            y = height - margin

    # add second page
    add_page()
    y = height - margin

    # Confusion Matrix
    y = draw_text(margin, y - 10, "Confusion Matrix:")
    y = draw_text(margin, y, str(conf_matrix))

    # ROC Curve
    plt.figure(figsize=(10, 7))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(unique_gestures), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_img_path = "roc_curve.png"
    plt.savefig(roc_img_path)
    plt.close()
    y = draw_text(margin, y - 10, "ROC Curve:")
    y = draw_image(roc_img_path, margin, y - 350, width - 2 * margin, 350)

    # Accuracy and Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    accuracy_img_path = "accuracy_curve.png"
    plt.savefig(accuracy_img_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    loss_img_path = "loss_curve.png"
    plt.savefig(loss_img_path)
    plt.close()

    add_page()
    y = height - margin

    y = draw_text(margin, y - 10, "Accuracy Curve:")
    y = draw_image(accuracy_img_path, margin, y - 350, width - 2 * margin, 350)

    add_page()
    y = height - margin

    y = draw_text(margin, y - 10, "Loss Curve:")
    y = draw_image(loss_img_path, margin, y - 350, width - 2 * margin, 350)

    y = draw_text(margin, y - 10, "End of Report - Generated by TrainAndEvaluateModel.py - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    c.save()
    buffer.seek(0)

    report_path = report_directory + "report_" + model_name.replace(".keras", ".pdf").replace("Modelos/", "")

    with open(report_path, "wb") as f:
        f.write(buffer.getvalue())

    # Remove the temporary images
    #os.remove("gesture_count.png")
    os.remove("roc_curve.png")
    os.remove("loss_curve.png")
    os.remove("accuracy_curve.png")


print("Treinar novo modelo ou avaliar um modelo existente? (t/a)")
choice = input()

if choice == 't' or choice == 'T':
    prefix = "Modelos/" + input("Enter the model prefix name: ")
    MODEL = prefix + ".keras"
    if not os.path.exists(MODEL):
        n_models = int(input("Number of models to train: "))
        if n_models == 1:
            EPOCH = int(input("Epochs: "))
            BATCH = int(input("Batch size: "))
            LEARNING_RATE = float(input("Learning rate: "))
        else:
            EPOCH = [0] * n_models
            BATCH = [0] * n_models
            LEARNING_RATE = [0] * n_models
            for i in range(n_models):
                print("Model ", i + 1)
                EPOCH[i] = int(input("Epochs: "))
                BATCH[i] = int(input("Batch size: "))
                LEARNING_RATE[i] = float(input("Learning rate: "))


        def create_model(EPOCH, BATCH, LEARNING_RATE):
            model = Sequential()
            model.add(Conv1D(64, 3, activation='relu', input_shape=(shape[1], shape[2])))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Conv1D(128, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(3))
            model.add(Dropout(0.3))
            model.add(Conv1D(256, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(3))
            model.add(Dropout(0.3))
            model.add(Conv1D(512, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(3))
            model.add(Flatten())
            model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.5))
            model.add(Dense(unique_gestures, activation='softmax'))

            optimizer = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

            history = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCH, validation_data=(x_val, y_val),
                                callbacks=[reduce_lr, early_stopping], verbose=1)
            return model, history


        if n_models == 1:
            model, history = create_model(EPOCH, BATCH, LEARNING_RATE)
            accuracy, fscore, fpr, tpr, roc_auc, class_report, conf_matrix = evaluate_model(model, x_val, y_val)
            print("Accuracy score: ", accuracy)
            model_name = build_model_name(prefix, str(EPOCH), str(BATCH), str(LEARNING_RATE), str(accuracy))
            report_directory = "Reports/"
            model.save(model_name)
            print("Model saved")
            print("Model Accuracy: ", accuracy)
            generate_report(model_name, accuracy, fscore, fpr, tpr, roc_auc, history, class_report, conf_matrix)
            print("Report generated")
        else:
            for i in range(n_models):
                print("Model ", i + 1)
                model, history = create_model(EPOCH[i], BATCH[i], LEARNING_RATE[i])
                accuracy, fscore, fpr, tpr, roc_auc, class_report, conf_matrix = evaluate_model(model, x_val, y_val)
                model_name = build_model_name(prefix, str(EPOCH[i]), str(BATCH[i]), str(LEARNING_RATE[i]),
                                              str(accuracy))
                report_directory = "Reports/"
                model.save(model_name)
                print("Model saved")
                generate_report(model_name, accuracy, fscore, fpr, tpr, roc_auc, history, class_report, conf_matrix)

    elif os.path.exists(MODEL):
        model = load_model(MODEL)
        print("Model already exists. Model loaded")
        accuracy, fscore, fpr, tpr, roc_auc, class_report, conf_matrix = evaluate_model(model, x_val, y_val)
    else:
        print("Model not found")
        exit()

elif choice == 'a' or choice == 'A':
    MODEL = "Modelos/" + input("Enter the model full name: ") + ".keras"
    if os.path.exists(MODEL):
        model = load_model(MODEL)
        print("Model loaded")
        accuracy, fscore, fpr, tpr, roc_auc, class_report, conf_matrix = evaluate_model(model, x_val, y_val)
        print(f"Accuracy: {accuracy:.4f}\nF-Score: {fscore:.4f}\nROC AUC:\n {roc_auc}\nConfusion Matrix:\n {conf_matrix}")
    else:
        print("Model not found")
        exit()
else:
    print("Invalid choice")
    exit()
