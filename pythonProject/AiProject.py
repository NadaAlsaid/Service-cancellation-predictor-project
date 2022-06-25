from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import *
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
data = pd.read_csv("CustomersDataset.csv")
# convert to numirecal data
data['gender'] = pd.factorize(data.gender)[0]  # 0=>Female     1=>Male
data['Dependents'] = pd.factorize(data.Dependents)[0]  # 0=>no   1=>yes
data['Partner'] = pd.factorize(data.Partner)[0]  # 0=>yes   1=>no
data['PhoneService'] = pd.factorize(data.PhoneService)[0]  # 0=>no   1=>yes
data['MultipleLines'] = pd.factorize(data.MultipleLines)[0]  # 0=>no   1=>yes
data['OnlineSecurity'] = pd.factorize(data.OnlineSecurity)[0]  # 0=>no   1=>yes
data['OnlineBackup'] = pd.factorize(data.OnlineBackup)[0]  # 0=>yes   1=>no
data['DeviceProtection'] = pd.factorize(data.DeviceProtection)[
                                        0]  # 0=>no   1=>yes
data['TechSupport'] = pd.factorize(data.TechSupport)[0]  # 0=>no   1=>yes
data['StreamingTV'] = pd.factorize(data.StreamingTV)[0]  # 0=>no   1=>yes
data['StreamingMovies'] = pd.factorize(data.StreamingMovies)[
                                       0]  # 0=>no   1=>yes
data['PaperlessBilling'] = pd.factorize(data.PaperlessBilling)[
                                        0]  # 0=>yes  1=>no
data['InternetService'] = pd.factorize(data.InternetService)[
                                       0]  # 0=>DSL  1=>Fiber   2=>No
# 0=>month_to_one      1=>one year      2=>two year
data['Contract'] = pd.factorize(data.Contract)[0]
# 0=>electronic   1=>Mailed   2=>Bank  3=>credit card
data['PaymentMethod'] = pd.factorize(data.PaymentMethod)[0]
data['Churn'] = pd.factorize(data.Churn)[0]  # 0=>No  1=>Yes
# delete customer id unwanted feature
data['customerID'].drop_duplicates(inplace=True)
data.drop('customerID', axis=1, inplace=True)

# find and drop null values
# print("before\n")
print(data.shape)
data.dropna(inplace=True)
# print("after\n")
# print(data.shape)
# print(data.isnull().sum())

# no null valuse

# change data type of total charge
# print(data.dtypes)
data['TotalCharges'] = data['TotalCharges'].astype(float)
# print(data['TotalCharges'].dtypes)
# print("#####################################")

# remove outliers


def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (1.5) * iqr
    upper = q3 + (1.5) * iqr
    ls = df.index[(df[ft] < lower) | (df[ft] > upper)]
    return ls


index_list = []
for feature in ['Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                'InternetService', 'PaymentMethod', 'Churn', 'MonthlyCharges', 'tenure', 'gender', 'Partner', 'SeniorCitizen', 'Contract',
                'TotalCharges']:  # , 'tenure''gender',, 'Partner', 'SeniorCitizen', 'Contract'
 index_list.extend(outliers(data, feature))
# print(index_list)


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


# print(data.shape)
data_new = remove(data, index_list)
# print(data_new.shape)

# data scaling
x = data_new.iloc[:, 0:19].values
y = data_new.iloc[:, 19].values
scaler = StandardScaler()
scaler.fit(data_new)


'''scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
x = scaler.fit_transform(x)'''


# split data 20% test and 80% training
X_train, X_test, y_train, y_test = train_test_split(data_new.drop(
    'Churn', axis=1), data_new['Churn'], test_size=0.3, random_state=0)


########################################################################################################################

# naive bayes
def naive_bayes(X_train, X_test):
    obj = GaussianNB()
    obj.fit(X_train, y_train)
    pre = obj.predict(X_test)
    accuracyTrainNaive = obj.score(X_train, y_train)
    cm_bayes = confusion_matrix(y_test, pre)
    accuracy_testNaive = metrics.accuracy_score(y_test, pre)

    return accuracyTrainNaive, cm_bayes, accuracy_testNaive


def TestNaive(cm_bayes, accuracy_testNaive):
 print('Accuracy Score of Naive Bayes (Test) is : ', accuracy_testNaive)
 print('confusion_matrix of Naive Bayes is : ', cm_bayes)
 print(sns.heatmap(cm_bayes))
 plt.show()


def TrainNaive(accuracyTrainNaive):
 print('Accuracy Score of Naive Bayes (Train) is : ', accuracyTrainNaive)


# TestNaive()
# TrainNaive()
##########################################################################################################################

# logistic
def logistic(X_train, X_test):
    modell = LogisticRegression(solver="liblinear", C=10.0, random_state=0)
    modell.fit(X_train, y_train)
    Prediction_test = modell.predict(X_test)
    accuracyTrainLog = modell.score(X_train, y_train)

    cm_logistic = confusion_matrix(y_test, Prediction_test)
    accuracyTestLog = metrics.accuracy_score(y_test, Prediction_test)
    print(classification_report(y_test, Prediction_test))
    return accuracyTrainLog, accuracyTestLog, cm_logistic


def TestLog(accuracyTestLog, cm_logistic):
 print("Accuracy Score of Logistic Regression (Test) is :", accuracyTestLog)
 print('confusion_matrix of Logistic Regression is : ', cm_logistic)
 # print(sns.heatmap(cm_logistic ))
 plt.show()


def TrainLog(accuracyTrainLog):
 print("Accuracy Score of Logistic Regression (Train) is :", accuracyTrainLog)

# TestLog()
# TrainLog()
#########################################################################################################################


# id3
dtc = DecisionTreeClassifier(max_depth=3, random_state=0)


def id3(X_train, X_test):

    dtc.fit(X_train, y_train)
    y_pred_dtc = dtc.predict(X_test)
    accuracyTrainId = dtc.score(X_train, y_train)
    cm_Id = confusion_matrix(y_test, y_pred_dtc)
    acc_dtc = accuracy_score(y_test, y_pred_dtc)

    return accuracyTrainId, cm_Id, acc_dtc


def TestId(cm_Id, acc_dtc):
 print("Accuracy Score of Id3 (Test) is :", acc_dtc)
 print('confusion_matrix of Id3 is : ', cm_Id)
 print(sns.heatmap(cm_Id))
 plt.show()


def TrainId(accuracyTrainId):
 print("Accuracy Score of Decision Tree is : ", accuracyTrainId)  # {acc_dtc}


'''dtc = DecisionTreeClassifier(max_depth=3, random_state=33)
dtc.fit(X_train, y_train)
# PREDICTION
y_pred_dtc = dtc.predict(X_test)
accuracyTrainId = dtc.score(X_train, y_train)
cm_Id = confusion_matrix(y_test, y_pred_dtc)
acc_dtc = accuracy_score(y_test, y_pred_dtc)

# IMPORTANR FEATURE
important = dtc.feature_importances_
Names = list(X_train.columns.values)
# for i in range (X_train.shape[1]):
# print(Names[i]," ---> ", important[i]*100,"%")


# DATA GRAPH
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=1000)
tree.plot_tree(dtc, feature_names=Names, filled=True);fig.savefig('Design.png')
fig.savefig('Design.png')


def TestId():
    print("Accuracy Score of Id3 (Test) is :", acc_dtc)
    print('confusion_matrix of Id3 is : ', cm_Id)
    # print(sns.heatmap(cm_logistic ))
    # plt.show()


def TrainId():
    print(f"Accuracy Score of Decision Tree is : ",
          accuracyTrainId)  # {acc_dtc}'''


# TestId()
# TrainId()
#########################################################################################################################

# svm

def Svm(X_train, X_test):
    # classi = SVC(kernel='linear')
    classi = svm.SVC()
    classi.fit(X_train, y_train)
    prediction = classi.predict(X_test)
    training_data_accuray = classi.score(X_train, y_train)
    test_data_accuray = accuracy_score(prediction, y_test)
    cm_svm = confusion_matrix(y_test, prediction)

    # plt.plot(x0,gutter_up, "k--" , linewidth=2)
    # plt.plot(x0,gutter_down ,"k--", linewidth=2)
    return training_data_accuray, test_data_accuray, cm_svm


def Testsvm(test_data_accuray, cm_svm):
 print('Accuracy on test data of SVM : ', test_data_accuray)
 print('confusion_matrix of SVM is : ', cm_svm)
 # print(sns.heatmap(cm_logistic ))
 # plt.show()
 from sklearn.datasets._samples_generator import make_blobs

 # creating datasets X containing n_samples
 # Y containing two classes
 X, Y = make_blobs(n_samples=2000, centers=2,
                   random_state=0, cluster_std=0.40)

 plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='summer');
 plt.show()


def Trainsvm(training_data_accuray):
 print('Accuracy on training data of SVM: ', training_data_accuray)

#########################################################################################################################

# randomForest


def randomForest(X_train, X_test):
    classifier = RandomForestClassifier(
        n_estimators=200, criterion='entropy', max_depth=9, random_state=0)
    classifier.fit(X_train, y_train)
    '''from sklearn.tree import plot_tree
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               cv = 4,
                               n_jobs=-1, verbose=1, scoring="accuracy")

    grid_search.fit(X_train, y_train)

    Names = list(X_train.columns.values)
    rf_best = grid_search.best_estimator_
    plt.figure(figsize=(80,40))
    plot_tree(rf_best.estimators_[200], feature_names = Names,filled=True)
    y_pred = grid_search.predict(X_test)'''
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acurracy_randomForestTrain = classifier.score(X_train, y_train)
    acurracy_randomForestTest = accuracy_score(y_test, y_pred)
    return cm, acurracy_randomForestTest, acurracy_randomForestTrain


'''for j in range(0 ,100):
        classifier = RandomForestClassifier(
            n_estimators = 100, criterion = 'gini')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print('RF for number of Trees : ' , j , ' is : \n' , cm)
        print('The Score is : ',classifier.score(X_test , y_test))
        print('=======================================================')
'''


def randomForestTest(cm, acurracy_randomForestTest):
   print('Accuracy on test data of Random Forest :', acurracy_randomForestTest)
   print('confusion matrix', cm)
   print(sns.heatmap(cm, center=True))
   plt.show()


def randomForesttrain(acurracy_randomForestTrain):
  print('Accuracy on train data of Random Forest : ', acurracy_randomForestTrain)

#########################################################################################################################
# KNN


def Knn(X_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    pre_knn = knn.predict(X_test)
    accuracyTrain_KNN = knn.score(X_train, y_train)
    cm_KNN = confusion_matrix(y_test, pre_knn)
    accuracy_test_KNN = accuracy_score(y_test, pre_knn)
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
    return accuracyTrain_KNN, accuracy_test_KNN, cm_KNN, train_accuracy, test_accuracy, neighbors


def TestKNN(accuracy_test_KNN, cm_KNN, train_accuracy, test_accuracy, neighbors):
 print('Accuracy Score of KNN (Test) is : ', accuracy_test_KNN)
 print('confusion_matrix of KNN is : ', cm_KNN)
 # print(sns.heatmap(cm_KNN))
 # plt.show()
 plt.plot(neighbors, test_accuracy, label='Testing KNN Accuracy')
 plt.plot(neighbors, train_accuracy, label='Training KNN Accuracy')

 plt.legend()
 plt.xlabel('neighbors')
 plt.ylabel('Accuracy')
 plt.show()


def TrainKNN(accuracyTrain_KNN):
 print('Accuracy Score of KNN (Train) is : ', accuracyTrain_KNN)


'''sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

'''from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_'''


##########################################################################################################################
#########################################################################################################################
# GUI
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk



global count , flage , our_images
flage = True

count = 0
pro = Tk()
pro.geometry('1250x750+180+30')
pro.resizable(False, False)
pro.title('Service cancellation predictor')
pro.iconbitmap('presentation.ico')
b = Image.open("2.jpg")

resized = b.resize((1255, 800), Image.ANTIALIAS)
    
new_image = ImageTk.PhotoImage(resized)
bb = Image.open("3.jpg")
resized1 = bb.resize((1250, 810), Image.ANTIALIAS)
    
new_image1 = ImageTk.PhotoImage(resized1)
bbbb = Image.open("6.png")
resized4 = bbbb.resize((1250, 810), Image.ANTIALIAS)
    
new_image4 = ImageTk.PhotoImage(resized4)
bbb = Image.open("1.jpg")
resized2 = bbb.resize((1250, 810), Image.ANTIALIAS)
    
new_image2 = ImageTk.PhotoImage(resized2)
our_images = [
        new_image4,
    	new_image2,
        new_image1,
]

my_canvas = Canvas(pro, width=1250, height=960, highlightthickness=0)
my_canvas.pack(fill="both", expand=True)
my_canvas.place(x=0, y=0)



def nextnext():
    

        my_canvas = Canvas(pro, width=1250, height=750, highlightthickness=0)
        my_canvas.pack(fill="both", expand=True)
        my_canvas.create_image(0, 0, image=new_image, anchor='nw')
        my_canvas.place(x=0, y=0)        
        v = IntVar()
        r1 = ttk.Radiobutton( text='Logistic regression', value=1, variable=v)
        
        r1.place(x=330, y=50)
        
        r2 = ttk.Radiobutton( text='SVM', value=2, variable=v)
        
        r2.place(x=480, y=50)
        
        r3 = ttk.Radiobutton( text='ID3', value=3, variable=v)
        
        r3.place(x=553, y=50)
        
        r4 = ttk.Radiobutton( text='Randomm Forest', value=4, variable=v)
        
        r4.place(x=620, y=50)
        
        r5 = ttk.Radiobutton( text='Navie Bytes', value=5, variable=v)
        r5.place(x=760, y=50)
        
        r6 = ttk.Radiobutton(text='KNN' , value=6, variable=v)
        r6.place(x=870, y=50)

        def Extraction():
            pr = Tk()
            pr.geometry('300x180+700+300')
            pr.resizable(False, False)
            pr.minsize(10, 10)
            pr.configure(bg='#004a87')
            pr.iconbitmap('presentation.ico')
            lda = LDA(n_components=1)
            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.transform(X_test)
            lbla = Label(pr, text="   After Feature Extraction", fg='white',
                         bg='#004a87', font=('Helvetic', 17, 'bold'))
            lbla.place(x=0, y=10)
            if v.get() == 1:
                accuracyTrainLog, accuracyTestLog, cm_logistic = logistic(
                    x_train, x_test)
                TrainLog(accuracyTrainLog)
                TestLog(accuracyTestLog, cm_logistic)
                pr.title('accuracyTrain of Logistic')
        
                lbltr = Label(pr, text="Logistic Train Accuracy", fg='black',
                              bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=35, y=40)
                lbltr1 = Label(pr, text=accuracyTrainLog, fg='white',
                               bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=70, y=70)
                pr.title('Logistic Test Accuracy')
                lbl = Label(pr, text="Logistic Test Accuracy", fg='black',
                            bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=35, y=100)
                lbl1 = Label(pr, text=accuracyTestLog, fg='white',
                             bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbl1.place(x=70, y=130)
            elif v.get() == 2:
                training_data_accuray, test_data_accuray, cm_svm = Svm(x_train, x_test)
                Trainsvm(training_data_accuray)
                Testsvm(test_data_accuray, cm_svm)
                pr.title('accuracyTrain of SVM')
        
                lbltr = Label(pr, text="SVM Train Accuracy", fg='black',
                              bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=50, y=40)
                lbltr1 = Label(pr, text=training_data_accuray, fg='white',
                               bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=70, y=70)
                pr.title('accuracyTest of SVM')
                lbl = Label(pr, text="SVM Test Accuracy", fg='black',
                            bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=50, y=100)
                lbl1 = Label(pr, text=test_data_accuray, fg='white',
                             bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbl1.place(x=67, y=130)
            elif v.get() == 3:
                accuracyTrainId, cm_Id, acc_dtc = id3(x_train, x_test)
                TrainId(accuracyTrainId)
                TestId(cm_Id, acc_dtc)
                pr.title('accuracyTrain of ID3')
        
                lbltr = Label(pr, text="ID3 Train Accuracy", fg='black',
                              bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=63, y=40)
                lbltr1 = Label(pr, text=accuracyTrainId, fg='white',
                               bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=70, y=70)
                pr.title('ID3 Test Accuracy')
                lbl = Label(pr, text="ID3 Train Accuracy", fg='black',
                            bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=63, y=100)
                lbl1 = Label(pr, text=acc_dtc, fg='white', bg='#004a87',
                             font=('Helvetic', 13, 'bold'))
                lbl1.place(x=70, y=130)
            elif v.get() == 4:
                cm, acurracy_randomForestTest, acurracy_randomForestTrain = randomForest(
                    x_train, x_test)
                randomForesttrain(acurracy_randomForestTrain)
                randomForestTest(cm, acurracy_randomForestTest)
                pr.title('accuracyTrain of Random Forest')
                lbltr = Label(pr, text="Random Forest Train Accuracy",
                              fg='black', bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=0, y=40)
                lbltr1 = Label(pr, text=acurracy_randomForestTrain,
                               fg='white', bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=70, y=70)
                pr.title('Random Forest Test Accuracy')
                lbl = Label(pr, text="Random Forest Test Accuracy",
                            fg='black', bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=3, y=100)
                lbl1 = Label(pr, text=acurracy_randomForestTest, fg='white',
                             bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbl1.place(x=130, y=130)
            elif v.get() == 5:
                accuracyTrainNaive, cm_bayes, accuracy_testNaive = naive_bayes(
                    x_train, x_test)
                TrainNaive(accuracyTrainNaive)
                TestNaive(cm_bayes, accuracy_testNaive)
                TrainNaive(accuracyTrainNaive)
                pr.title('accuracyTrain of Naive')
        
                lbltr = Label(pr, text="Naive Train Accuracy", fg='black',
                              bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=48, y=40)
                lbltr1 = Label(pr, text=accuracyTrainNaive, fg='white',
                               bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=70, y=70)
                pr.title('accuracyTest of Naive')
        
                lbl = Label(pr, text="Naive Test Accuracy", fg='black',
                            bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=48, y=100)
                lbl1 = Label(pr, text=accuracy_testNaive, fg='white',
                             bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbl1.place(x=70, y=130)
            elif v.get() == 6:
                accuracyTrain_KNN, accuracy_test_KNN, cm_KNN,  train_accuracy, test_accuracy, neighbors = Knn(
                    x_train, x_test)
                TrainKNN(accuracyTrain_KNN)
                TestKNN(accuracy_test_KNN, cm_KNN,
                        train_accuracy, test_accuracy, neighbors)
                pr.title('accuracyTrain of KNN')
                lbltr = Label(pr, text="KNN Train Accuracy", fg='black',
                              bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbltr.place(x=53, y=40)
                lbltr1 = Label(pr, text=accuracyTrain_KNN, fg='white',
                               bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbltr1.place(x=67, y=70)
        
                pr.title('accuracyTest of KNN')
                lbl = Label(pr, text="KNN Test Accuracy", fg='black',
                            bg='#004a87', font=('Helvetic', 14, 'bold'))
                lbl.place(x=53, y=100)
                lbl1 = Label(pr, text=accuracy_test_KNN, fg='white',
                             bg='#004a87', font=('Helvetic', 13, 'bold'))
                lbl1.place(x=67, y=130)
            pr.mainloop()

        def Train():
                pr = Tk()
                pr.geometry('300x180+700+300')
                pr.resizable(False, False)
                pr.configure(bg='#004a87')
                pr.iconbitmap('presentation.ico')
                if v.get() == 1:
                    accuracyTrainLog, accuracyTestLog, cm_logistic = logistic(
                        X_train, X_test)
                    TrainLog(accuracyTrainLog)
                    pr.title('accuracyTrain of Logistic')
        
                    lbl = Label(pr, text="Logistic Train Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=35, y=60)
                    lbl1 = Label(pr, text=accuracyTrainLog, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=65, y=90)
        
                elif v.get() == 2:
        
                    training_data_accuray, test_data_accuray, cm_svm = Svm(
                        X_train, X_test)
                    Trainsvm(training_data_accuray)
                    pr.title('accuracyTrain of SVM')
        
                    lbl = Label(pr, text="SVM Train Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=50, y=60)
        
                    lbl1 = Label(pr, text=training_data_accuray, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=70, y=90)
        
                elif v.get() == 3:
                    accuracyTrainId, cm_Id, acc_dtc = id3(X_train, X_test)
                    pr.title('accuracyTrain of ID3')
        
                    lbl = Label(pr, text="ID3 Train Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=63, y=60)
        
                    lbl1 = Label(pr, text=accuracyTrainId, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=77, y=90)
        
                    important = dtc.feature_importances_
                    Names = list(X_train.columns.values)
                    for i in range(X_train.shape[1]):
                        print(Names[i], " ---> ", important[i]*100, "%")
            
                    # DATA GRAPH
                    '''plt.rcParams.update({'font.size': 10})
                    fig, axes = plt.subplots(
                        nrows=1, ncols=1, figsize=(4, 4), dpi=1000)
                    tree.plot_tree(dtc, feature_names=Names, filled=True)
                    fig.savefig('Design.png')'''
        
                    TrainId(accuracyTrainId)
                elif v.get() == 4:
                    cm, acurracy_randomForestTest, acurracy_randomForestTrain = randomForest(
                        X_train, X_test)
                    randomForesttrain(acurracy_randomForestTrain)
                    pr.title('accuracyTrain of Random Forest')
                    lbl = Label(pr, text="Random Forest Train Accuracy",
                                fg='black', bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=0, y=60)
        
                    lbl1 = Label(pr, text=acurracy_randomForestTrain,
                                 fg='white', bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=70, y=90)
        
                elif v.get() == 5:
                    accuracyTrainNaive, cm_bayes, accuracy_testNaive = naive_bayes(
                        X_train, X_test)
                    TrainNaive(accuracyTrainNaive)
                    pr.title('accuracyTrain of Naive')
        
                    lbl = Label(pr, text="Naive Train Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=48, y=60)
        
                    lbl1 = Label(pr, text=accuracyTrainNaive, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=70, y=90)
        
                elif v.get() == 6:
                    accuracyTrain_KNN, accuracy_test_KNN, cm_KNN,  train_accuracy, test_accuracy, neighbors = Knn(
                        X_train, X_test)
                    TrainKNN(accuracyTrain_KNN)
                    pr.title('accuracyTrain of KNN')
        
                    lbl = Label(pr, text="KNN Train Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=53, y=60)
        
                    lbl1 = Label(pr, text=accuracyTrain_KNN, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=67, y=90)
        
                pr.mainloop()
        
        def Test():
                pr = Tk()
                pr.geometry('300x180+700+300')
                pr.resizable(False, False)
                pr.configure(bg='#004a87')
                pr.iconbitmap('presentation.ico')
                pr.minsize(10, 10)
                if v.get() == 1:
                    accuracyTrainLog, accuracyTestLog, cm_logistic = logistic(
                        X_train, X_test)
                    TestLog(accuracyTestLog, cm_logistic)
                    pr.title(' Logistic Test Accuracy')
                    
                    lbl = Label(pr, text="Logistic Test Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=35, y=60)
                    lbl1 = Label(pr, text=accuracyTestLog, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=65, y=90)
                    
                elif v.get() == 2:
                    training_data_accuray, test_data_accuray, cm_svm = Svm(
                        X_train, X_test)
                    Testsvm(test_data_accuray, cm_svm)
                    pr.title('accuracyTest of SVM')
                    lbl = Label(pr, text="SVM Test Accuracy ", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=50, y=60)
                    lbl1 = Label(pr, text=test_data_accuray, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=67, y=90)
                elif v.get() == 3:
                    accuracyTrainId, cm_Id, acc_dtc = id3(X_train, X_test)
                    TestId(cm_Id, acc_dtc)
                    pr.title('accuracyTest of ID3')
                    lbl = Label(pr, text="ID3 Test Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=63, y=60)
                    lbl1 = Label(pr, text=acc_dtc, fg='white',
                                  bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=70, y=90)
                elif v.get() == 4:
                    cm, acurracy_randomForestTest, acurracy_randomForestTrain = randomForest(
                        X_train, X_test)
                    randomForestTest(cm, acurracy_randomForestTest)
                    pr.title('Random Forest Test Accuracy')
                    lbl = Label(pr, text="Random Forest Test Accuracy",
                                fg='black', bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=3, y=60)
                    lbl1 = Label(pr, text=acurracy_randomForestTest, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=70, y=90)
                elif v.get() == 5:
                    accuracyTrainNaive, cm_bayes, accuracy_testNaive = naive_bayes(
                        X_train, X_test)
                    TestNaive(cm_bayes, accuracy_testNaive)
                    pr.title('accuracyTest of Naive')
        
                    lbl = Label(pr, text=" Naive Test Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=48, y=60)
                    lbl1 = Label(pr, text=accuracy_testNaive, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=75, y=90)
                elif v.get() == 6:
                    accuracyTrain_KNN, accuracy_test_KNN, cm_KNN, train_accuracy, test_accuracy, neighbors = Knn(
                        X_train, X_test)
                    TestKNN(accuracy_test_KNN, cm_KNN,
                            train_accuracy, test_accuracy, neighbors)
                    pr.title('KNN Test Accuracy')
        
                    lbl = Label(pr, text="KNN Test Accuracy", fg='black',
                                bg='#004a87', font=('Helvetic', 15, 'bold'))
                    lbl.place(x=53, y=60)
                    lbl1 = Label(pr, text=accuracy_test_KNN, fg='white',
                                 bg='#004a87', font=('Helvetic', 13, 'bold'))
                    lbl1.place(x=67, y=90)
                pr.mainloop()
        
        def prediction():
                n1 = en1.get()
                Label(pro, text=n1)
        
                n2 = en2.get()
                if n2.casefold() == 'yes':
                    n2 = 0
                elif n2.casefold() == 'no':
                    n2 = 1
                Label(pro, text=n2)
        
                n3 = en3.get()
                if n3.casefold() == 'yes':
                    n3 = 1
                elif n3.casefold() == 'no':
                    n3 = 0
                Label(pro, text=n3)
        
                n4 = en4.get()
                if n4.casefold() == 'yes':
                    n4 = 1
                elif n4.casefold() == 'no':
                    n4 = 0
                Label(pro, text=n4)
        
                n5 = en5.get()
                if n5.casefold() == 'yes':
                    n5 = 1
                elif n5.casefold() == 'no':
                    n5 = 0
                Label(pro, text=n5)
        
                n6 = en6.get()
                if n6.casefold() == 'month-to-month':
                    n6 = 0
                elif n6.casefold() == 'one year':
                    n6 = 1
                elif n6.casefold() == 'two year':
                    n6 = 2
                Label(pro, text=n6)
        
                n7 = en7.get()
                Label(pro, text=n7)
        
                n8 = en8.get()
                if n8.casefold() == 'female':
                    n8 = 0
                elif n8.casefold() == 'male':
                    n8 = 1
                Label(pro, text=n8)
        
                n9 = en9.get()
                if n9.casefold() == 'yes':
                    n9 = 1
                elif n9.casefold() == 'no':
                    n9 = 0
                Label(pro, text=n9)
        
                n10 = en10.get()
                if n10.casefold() == 'yes':
                    n10 = 1
                elif n10.casefold() == 'no':
                    n10 = 0
                Label(pro, text=n10)
        
                n11 = en11.get()
                if n11.casefold() == 'yes':
                    n11 = 0
                elif n11.casefold() == 'no':
                    n11 = 1
        
                Label(pro, text=n11)
        
                n12 = en12.get()
                if n12.casefold() == 'yes':
                    n12 = 1
                elif n12.casefold() == 'no':
                    n12 = 0
        
                Label(pro, text=n12)
        
                n13 = en13.get()
                if n13.casefold() == 'yes':
                    n13 = 0
                elif n13.casefold() == 'no':
                    n13 = 1
                Label(pro, text=n13)
        
                n14 = en14.get()
                Label(pro, text=n14)
        
                n15 = en15.get()
                Label(pro, text=n15)
        
                n16 = en16.get()
                Label(pro, text=n16)
        
                n17 = en17.get()
                if n17.casefold() == 'dsl':
                    n17 = 0
                elif n17.casefold() == 'fiber optic':
                    n17 = 1
                elif n17.casefold() == 'no':
                    n17 = 2
                Label(pro, text=n17)
        
                n18 = en18.get()
                if n18.casefold() == 'no':
                    n18 = 0
                elif n18.casefold() == 'yes':
                    n18 = 1
        
                Label(pro, text=n18)
        
                n19 = en19.get()
                if n19.casefold() == 'no':
                    n19 = 0
                elif n19.casefold() == 'yes':
                    n19 = 1
        
                Label(pro, text=n19)
        
                n20 = en20.get()
                if n20.casefold() == 'electronic check':
                    n20 = 0
                elif n20.casefold() == 'mailed check':
                    n20 = 1
                elif n20.casefold() == 'bank transfer (automatic)':
                    n20 = 2
                elif n20.casefold() == 'credit card (automatic)':
                    n20 = 3
                Label(pro, text=n20)
        
                arr = [[n2, n3, n4, n5, n6, float(n7), n8, n9, n10, n11, n12, n13, float(
                    n14), int(n15), float(n16), n17, n18, n19, n20]]
               # print(n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20)int(n16)float(n7)float(n14)
               # print(arr)
        
                def predict_log():
                    modell = LogisticRegression(
                        solver="liblinear", C=10.0, random_state=0)
                    modell.fit(X_train, y_train)
                    Prediction_test = modell.predict(X_test)
                    accuracyTrainLog = modell.score(X_train, y_train)
        
                    cm_logistic = confusion_matrix(y_test, Prediction_test)
                    accuracyTestLog = metrics.accuracy_score(y_test, Prediction_test)
                    pr = modell.predict(arr)
                    if pr == 1:
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        prr.iconbitmap('presentation.ico')
                        print('The prediction with Logistic Regression is : Yes')
                        lbl1 = Label(prr, text='            The prediction with LogisticRegression is : Yes',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=6, y=80)
                        prr.mainloop()
                    elif pr == 0:
                        print('The prediction with Logistic Regression is : No')
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        lbl1 = Label(prr, text='             The prediction with Logistic Regression is : No',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=6, y=80)
                        prr.mainloop()
        
                def predict_Naive():
                    obj = GaussianNB()
                    obj.fit(X_train, y_train)
                    pre = obj.predict(X_test)
                    accuracyTrainNaive = obj.score(X_train, y_train)
                    cm_bayes = confusion_matrix(y_test, pre)
                    accuracy_testNaive = metrics.accuracy_score(y_test, pre)
                    pr = obj.predict(arr)
                    if pr == 1:
                            prr = Tk()
                            prr.geometry('500x180+580+300')
                            prr.resizable(False, False)
                            prr.minsize(100, 100)
                            prr.configure(bg='#004a87')
                            prr.title('Prediction')
                            prr.iconbitmap('presentation.ico')
                            print('           The prediction with Naive Bayes is : Yes')
                            lbl1 = Label(prr, text='                  The prediction with Naive Bayes is : Yes',
                                          fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                            lbl1.place(x=0, y=80)
                            prr.mainloop()
                    elif pr == 0:
                            print('The prediction with Naive Bayes is : No')
                            prr = Tk()
                            prr.geometry('500x180+580+300')
                            prr.resizable(False, False)
                            prr.minsize(100, 100)
                            prr.title('Prediction')
                            prr.iconbitmap('presentation.ico')
                            prr.configure(bg='#004a87')
                            lbl1 = Label(prr, text='                  The prediction with Naive Bayes is : No',
                                          fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                            lbl1.place(x=0, y=80)
                            prr.mainloop()
        
                def predict_dession():
                    dtc = DecisionTreeClassifier(max_depth=3, random_state=33)
                    dtc.fit(X_train, y_train)
                    y_pred_dtc = dtc.predict(X_test)
                    accuracyTrainId = dtc.score(X_train, y_train)
                    cm_Id = confusion_matrix(y_test, y_pred_dtc)
                    acc_dtc = accuracy_score(y_test, y_pred_dtc)
                    pr = dtc.predict(arr)
                    if pr == 1:
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        prr.iconbitmap('presentation.ico')
                        print('The prediction with Dession Tree is : Yes')
                        lbl1 = Label(prr, text='                  The prediction with Dession Tree is : Yes',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=0, y=80)
                        prr.mainloop()
                    elif pr == 0:
                        print('    The prediction with Dession Tree is : No')
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        lbl1 = Label(prr, text='                   The prediction with Dession Tree is : No',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=0, y=80)
                        prr.mainloop()
        
                def predict_svm():
                    classi = svm.SVC()
                    classi.fit(X_train, y_train)
                    prediction = classi.predict(X_test)
                    training_data_accuray = classi.score(X_train, y_train)
                    test_data_accuray = accuracy_score(prediction, y_test)
                    cm_svm = confusion_matrix(y_test, prediction)
                    pr = classi.predict(arr)
                    if pr == 1:
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.title('Prediction')
                        prr.iconbitmap('presentation.ico')
                        prr.configure(bg='#004a87')
                        print('The prediction with SVM is : Yes')
                        lbl1 = Label(prr, text='                 The prediction with SVM is : Yes',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=30, y=80)
                        prr.mainloop()
                    elif pr == 0:
                        prr = Tk()
                        prr.configure(bg='#004a87')
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.title('Prediction')
                        print('The prediction with SVM is : No')
                        lbl1 = Label(prr, text='                  The prediction with SVM is : No',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=30, y=80)
                        prr.mainloop()
        
                def predict_random():
                    classifier = RandomForestClassifier(
                        n_estimators=200, criterion='entropy', max_depth=10)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    acurracy_randomForestTrain = classifier.score(X_train, y_train)
                    acurracy_randomForestTest = accuracy_score(y_test, y_pred)
                    pr = classifier.predict(arr)
                    if pr == 1:
                        print('The prediction with Random Forset is : Yes')
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        lbl1 = Label(prr, text='                The prediction with Random Forset is : Yes',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=0, y=80)
                        prr.mainloop()
                    elif pr == 0:
                        print('The prediction with Random Forset is : No')
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.title('Prediction')
                        prr.configure(bg='#004a87')
                        lbl1 = Label(prr, text='                The prediction with Random Forset is : No',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=0, y=80)
                        prr.mainloop()
        
                def predict_knn():
                    knn = KNeighborsClassifier(n_neighbors=10)
                    knn.fit(X_train, y_train)
                    pre_knn = knn.predict(X_test)
                    accuracyTrain_KNN = knn.score(X_train, y_train)
                    cm_KNN = confusion_matrix(y_test, pre_knn)
                    accuracy_test_KNN = accuracy_score(y_test, pre_knn)
                    neighbors = np.arange(1, 9)
                    train_accuracy = np.empty(len(neighbors))
                    test_accuracy = np.empty(len(neighbors))
        
                    # Loop over K values
                    for i, k in enumerate(neighbors):
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train, y_train)
        
                        # Compute training and test data accuracy
                        train_accuracy[i] = knn.score(X_train, y_train)
                        test_accuracy[i] = knn.score(X_test, y_test)
                    pr = knn.predict(arr)
                    if pr == 1:
                        prr = Tk()
        
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.configure(bg='#004a87')
                        prr.title('Prediction')
                        prr.iconbitmap('presentation.ico')
                        print('The prediction with KNN is : Yes')
                        lbl1 = Label(prr, text='                         The prediction with KNN is : Yes',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.place(x=0, y=80)
                        lbl1.config(anchor=CENTER)
                        prr.mainloop()
                    elif pr == 0:
                        prr = Tk()
                        prr.geometry('500x180+580+300')
                        prr.resizable(False, False)
                        prr.minsize(100, 100)
                        prr.iconbitmap('presentation.ico')
                        prr.title('Prediction')
                        prr.configure(bg='#004a87')
                        print('The prediction with KNN is : No')
                        lbl1 = Label(prr, text='                         The prediction with KNN is : No',
                                     fg='black', bg='#004a87', font=('Helvetic', 13, 'bold'))
                        lbl1.config(anchor=CENTER)
                        lbl1.place(x=0, y=80)
                        prr.mainloop()
        
                if v.get() == 1:
                    predict_log()
                elif v.get() == 2:
                    predict_svm()
                elif v.get() == 3:
                    predict_dession()
                elif v.get() == 4:
                    predict_random()
                elif v.get() == 5:
                    predict_Naive()
                elif v.get() == 6:
                    predict_knn()
        
        bt1 = Button( text='Predict', fg='black', bg='white', width='25', height='2',
                         font=('Helvetic', 9, 'italic', 'bold'), activebackground='black', activeforeground='white', command=prediction)
        bt1.place(x=550, y=650)
        
        bt2 = Button( text='Train', fg='black', bg='white', width='20', height='1', font=('Helvetic', 9, 'italic', 'bold'),
                         activebackground='black', activeforeground='white', command=Train)
        bt2.place(x=330, y=85)
        bt3 = Button( text='Test', fg='black', bg='white', width='20', height='1', font=('Helvetic', 9, 'italic', 'bold'),
                         activebackground='black', activeforeground='white', command=Test)
        bt3.place(x=480, y=85)
        
        bt4 = Button( text='Accuracy after feature extraction', fg='black', bg='white', width='40', height='1', font=('Helvetic', 9, 'italic', 'bold'),
                         activebackground='black', activeforeground='white', command=Extraction)
        bt4.place(x=629, y=85)
        
        lbl1 = Label( text='Methodology', fg='white',
                         bg='#003e6f', font=(NONE,18 ,'bold'))  # grey
        lbl1.place(x=553, y=2)
        
        lbl2 = Label(pro, text='Customer Data', fg='white',
                         bg='#003e6f', font=(NONE, 17,'bold'))
        lbl2.place(x=20, y=112)
        
        
        lbl3 = Label(pro, text='CustomerID', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl3.place(x=30, y=190)
        
        en1 = Entry(pro, bg='silver')
        en1.place(x=180, y=193)
        
        lbl4 = Label(pro, text='Partner', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl4.place(x=30, y=240)
        
        en2 = Entry(pro, bg='silver')
        en2.place(x=180, y=243)
        lbl5 = Label(pro, text='PhoneService', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl5.place(x=30, y=295)
        en3 = Entry(pro, bg='silver')
        en3.place(x=180, y=298)
        lbl6 = Label(pro, text='OnlineSecurity', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl6.place(x=30, y=340)
        en4 = Entry(pro, bg='silver')
        en4.place(x=180, y=343)
        lbl7 = Label(pro, text='TechSupport', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl7.place(x=30, y=390)
        en5 = Entry(pro, bg='silver')
        en5.place(x=180, y=393)
        
        lbl8 = Label(pro, text='Contract', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl8.place(x=30, y=440)
        
        en6 = Entry(pro, bg='silver')
        en6.place(x=180, y=443)
        
        lbl9 = Label(pro, text='MonthlyCharges', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl9.place(x=30, y=490)
        
        en7 = Entry(pro, bg='silver')
        en7.place(x=180, y=493)
        
        lbl10 = Label(pro, text='Gender', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl10.place(x=490, y=190)
        
        en8 = Entry(pro, bg='silver')
        en8.place(x=640, y=193)
        lbl11 = Label(pro, text='Dependent', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl11.place(x=490, y=240)
        en9 = Entry(pro, bg='silver')
        en9.place(x=640, y=243)
        lbl12 = Label(pro, text='MultipleLines', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl12.place(x=490, y=290)
        
        en10 = Entry(pro, bg='silver')
        en10.place(x=640, y=293)
        lbl13 = Label(pro, text='OnlineBackup', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl13.place(x=490, y=340)
        en11 = Entry(pro, bg='silver')
        en11.place(x=640, y=343)
        
        lbl14 = Label(pro, text='StreamingTV', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl14.place(x=490, y=390)
        
        en12 = Entry(pro, bg='silver')
        en12.place(x=640, y=393)
        
        lbl15 = Label(pro, text='PaperlessBilling',
                          fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl15.place(x=490, y=440)
        
        en13 = Entry(pro, bg='silver')
        en13.place(x=640, y=443)
        lbl16 = Label(pro, text='TotalCharges', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl16.place(x=490, y=490)
        
        en14 = Entry(pro, bg='silver')
        en14.place(x=640, y=493)
        
        lbl17 = Label(pro, text='SeniorCitizen', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl17.place(x=940, y=190)
        en15 = Entry(pro, bg='silver')
        en15.place(x=1100, y=193)
        lbl18 = Label(pro, text='Tenure', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl18.place(x=940, y=240)
        
        en16 = Entry(pro, bg='silver')
        en16.place(x=1100, y=243)
        lbl19 = Label(pro, text='InternetService', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl19.place(x=940, y=290)
        en17 = Entry(pro, bg='silver')
        en17.place(x=1100, y=293)
        
        lbl20 = Label(pro, text='DeviceProtection',
                          fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl20.place(x=940, y=340)
        
        en18 = Entry(pro, bg='silver')
        en18.place(x=1100, y=343)
        lbl21 = Label(pro, text='StreamingMovies', fg='black', bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl21.place(x=940, y=390)
        
        en19 = Entry(pro, bg='silver')
        en19.place(x=1100, y=393)
        
        lbl22 = Label(pro, text='PaymentMethod', fg='black',  bg='#c3ecf3', font=(NONE, 13,'bold'))
        lbl22.place(x=940, y=440)
        
        en20 = Entry(pro, bg='silver')
        en20.place(x=1100, y=443)
       


def next():

         global count, flage, our_images

           # Set the canvas image

         if count == 3:
             my_canvas.delete('all')
             my_canvas.place(x=-1255, y=-990)
             nextnext()
                   
         else:
                
                my_canvas.create_image(
                    0, 0, image=our_images[count], anchor='nw')
                
                if count <2:
                    pro.after(2000, next)
                elif count == 2:
                    pro.after(4000, next)
                count += 1
                

next()

pro.mainloop()