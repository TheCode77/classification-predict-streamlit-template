"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
import streamlit as st
import joblib,os
import pandas as pd
import numpy as np
from PIL import Image
import string 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC

news_vectorizer = open("resources/vector.pickle","rb")
tweet_cv = joblib.load(news_vectorizer) 

raw = pd.read_csv("resources/train.csv")
	
def main():
	"""Tweet Classifier App with Streamlit """

	st.title("Tweet Classification")
	st.subheader("Climate change tweet classification")

	options = ["Home", "Prediction", "Information", "About Us", "Project: Summary", "Project: Process Descriptions", "Project: Team", "Exploratory Data Analysis", "What You Can Do"]
	st.sidebar.subheader("Welcome")
	st.sidebar.write("Welcome to our Streamlit App. This app has been proudly developed by Data Solutions, and its purpose is to educate the user about climate, whilst illustrating the various departments explored for the project that Databotics Solutions an Friends of The Earth had undergone.")
	selection = st.sidebar.selectbox("Choose Option", options)
	st.sidebar.subheader("Please find the company's details below:")
	st.sidebar.write("**Office Number:** +27-11-237-8901")
	st.sidebar.write("**Email Address** dbsolutions@gmail.com")
	st.sidebar.write("**Website Address:** www.dbsolutions.com")
	st.sidebar.write("**Headquarters Address:** 26 Cameron St., Cape Town, Western Cape, South Africa, 1259")



	# Building out the "What You Can Do" page
	if selection == "What You Can Do":
		st.title("What You Can Do")
		st.image("Images//PBM.cms")

		st.subheader("What is Plant-Based Meat?")
		st.write("Plant-based meat is just what it sounds like: a substitute for real meat that uses no animal products and is primarily made up of vegetables, grains, and legumes, tofu, tempeh, and seitan.")
		st.write("An example of this is Beyond Burger, the self-proclaimed “first plant-based burger designed to look, cook and taste like animal meat and sold in the meat section; this burger is created using peas, mung beans, and brown rice — known as a “complete protein” as it contains all nine essential amino acids.")
	
		st.header("Ingredients")
		st.subheader("Yummy Meatballs")
		st.image("Images//Meatballs.jpg")
		st.write("Products that are labelled “plant-based” generally don’t include animal products — although this isn’t necessarily guaranteed. A plant-based meat manufacturer’s first job is to gather ingredients that will both meet this criteria and mix well together. The foundational elements in plant-based meat are a protein of some sort (like tofu, tempeh, or soy), plant oils (like sunflower or canola oil), and a vegan binding agent (like gluten, aquafaba, or beans.")
		st.subheader("Recipe")
		st.write("- 1 (28 ounce) can whole peeled tomatoes, drained, juice reserved")
		st.write("- 4 tablespoons olive oil, divided")
		st.write("- 1 medium onion, finely chopped")
		st.write("- 3 cloves garlic, minced")
		st.write("- 1 (8 ounce) can tomato sauce")
		st.write("- 1 teaspoon dried oregano")
		st.write("- 1 ¼ teaspoons salt, divided")
		st.write("- ½ teaspoon freshly ground black pepper, divided")
		st.write("- 1 pound meatless ground beef substitute (such as Beyond Meat® Beyond Beef Plant-Based Ground)")
		st.write("- 1 tablespoon vegan bread crumbs")
		st.write("- 1 ½ teaspoons dried parsley flakes")
		st.write("- ¼ teaspoon garlic powder")
		st.write("- ¼ teaspoon onion powder")
		st.write("- 1 (12 ounce) package spaghetti")
		
		st.subheader("World's Best (Now Vegetarian!) Lasagna")
		st.image("Images//Lasagna.jpg")
		st.write("- **Prep Time:**")
		st.write("30 mins")
		st.write("- **Cook Time:**")
		st.write("2 hrs 30 mins")
		st.write("- **Additional Time:**")
		st.write("15 mins")
		st.write("- **Total Time:**")
		st.write("3 hrs 15 mins")
		st.write("- **Servings:**")
		st.write("12")
		st.write("- **Yield:**")
		st.write("1 lasagna")
		st.write("Jump to Nutrition Facts")
		st.subheader("Recipe")
		st.write("- 1 tablespoon olive oil")
		st.write("- ½ cup minced onion")
		st.write("- 2 cloves garlic, crushed")
		st.write("- 8 ounces plant-based hot Italian-style sausage (such as Beyond Meat®), chopped")
		st.write("- 6 ounces cooked and crumbled ground meat substitute (such as BOCA)")
		st.write("- 1 (15 ounce) can crushed tomatoes")
		st.write("- 1 (8 ounce) can tomato sauce")
		st.write("- 1 (6 ounce) can tomato paste")
		st.write("- ¼ cup water")
		st.write("- 3 tablespoons chopped fresh parsley, divided")
		st.write("- 1 tablespoon white sugar")
		st.write("- ¾ teaspoon dried basil")
		st.write("- 1 teaspoon salt, divided")
		st.write("- ½ teaspoon Italian seasoning")
		st.write("- ¼ teaspoon fennel seeds")
		st.write("- ⅛ teaspoon ground black pepper")
		st.write("- 12 lasagna noodles")
		st.write("- 1 egg, lightly beaten")
		st.write("- 1 (15 ounce) container ricotta cheese")
		st.write("- 12 ounces mozzarella cheese, sliced")
		st.write("- 6 ounces grated Parmesan cheese")
		st.write("- cooking spray")

		st.header("Local Farmers and Plant-Based Meat: Opportunities and Challenges")
		st.image("Images//Local Farm.webp")
		st.write("The rise of plant-based meat alternatives presents a significant opportunity and challenge for local farmers. While some may face potential disruptions to their traditional farming practices, others can seize the opportunity to become integral players in the burgeoning plant-based food industry.")
		st.subheader("Opportunities for Local Farmers:")
		st.write("- **Shifting production:** Many of the crops used in plant-based meat, such as soybeans, peas, and lentils, are already grown by local farmers. By shifting their focus to these crops, farmers can tap into a growing market with potentially higher profits.")
		st.write("- **Direct-to-consumer sales:** Farmers can sell directly to plant-based meat manufacturers, cutting out middlemen and increasing their profit margins.")
		st.write("- **Diversification:** Diversifying their crop portfolio with various plant-based meat ingredients can help farmers mitigate risk and stabilize their income.")
		st.write("- **Sustainable practices:** Plant-based meat production often requires less land and water than traditional meat production, aligning with sustainable agricultural practices and potentially attracting eco-conscious consumers.")
		st.write("- **Increased demand:** The global plant-based meat market is expected to reach $29.4 billion by 2032, presenting a significant opportunity for local farmers to increase their sales and expand their operations.")
		st.image("Images//One Farn.jpg")
		st.subheader("Challenges for Local Farmers:")
		st.write("- **Competition:** Local farmers may face competition from large-scale farms specializing in plant-based protein production.")
		st.write("- **Infrastructure:** Adapting existing farms or building new infrastructure to meet the specific needs of plant-based protein production may require significant investment.")
		st.write("- **Market volatility:** The plant-based meat market is still relatively new and may experience fluctuations in demand, making it challenging for farmers to predict future profitability.")
		st.write("- **Technical expertise:** Growing and processing crops for plant-based protein production may require specific technical knowledge and skills that some local farmers may not possess.")
		st.write("- **Access to resources:** Smaller farms may lack access to resources such as financing, technology, and market connections, hindering their ability to compete effectively.")
		st.image("Images//Farmers.webp")
		st.subheader("Strategies for Success:")
		st.write("- **Collaborations:** Farmers can form cooperatives or partnerships to share resources, knowledge, and expertise.")
		st.write("- **Education and training:** Participating in educational programs and training workshops can help farmers acquire the necessary skills and knowledge for plant-based protein production.")
		st.write("- **Vertical integration:** Building partnerships or integrating operations with plant-based meat manufacturers can provide farmers with stable markets and better prices for their crops.")
		st.write("- **Marketing and branding:** Local farmers can highlight the sustainability and ethical aspects of their products to differentiate themselves from large-scale producers and attract environmentally conscious consumers.")
		st.write("- **Government support:** Government policies and programs that promote plant-based protein production and support local farmers can play a crucial role in creating a more sustainable and equitable food system.")

		st.header("Plant-Based Meat Challenges: Motivating Change with No Meat Mondays and Beyond")
		st.image("Images//PBM.jpg")



	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): 
			st.write(raw[['sentiment', 'message']]) 


	# Building out the "Exploratory Data Analysis" page
	if selection == "Exploratory Data Analysis":
		st.title("Exploratory Data Analysis")
		st.image("Images//Exploratory.jpg")
		st.subheader("Data Description")
		st.write("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:")
		st.image("Images//Data Description.png")

		st.subheader("Sentiments Distribution")
		st.write("- When we look at the histogram of our sentiments, we can see that a vast majority of them are Pro (the tweet supports the belief of man-made climate change).")
		st.write("- There is a strong imbalance for sentiment classes")
		st.write("- Sentiment class '1' (Pro) dominates the chart with over 50% contribution, while class '-1' (Anti) lags behind with 8%.")
		st.write("- This tells us that majority of tweets believes in man-made climate change")
		st.image("Images//Sentiments Distribution.png")

		st.subheader("Five-number summary")
		st.write("Our target data, including the message feature, exhibits no instances of missing values. There is a notable imbalance among sentiment classes. Sentiment class '1' (Pro) significantly outweighs the others, contributing over 50%, while class '-1' (Anti) trails with an 8% contribution.")
		st.write("From the box plots below, we can see that tweets that fall in the pro climate change class are generally longer and the shortest tweets belong to the anti climate change class. We also notice that neutral climate change tweets tend to have the most variability in tweet length.")
		st.image("Images//Boxplots.png")

		st.subheader("Comparing Raw and Clean Data")
		st.image("Images//Raw Data.png")
		st.image("Images//Clean Data.png")
		st.subheader("Observation")
		st.write("**Raw Data Sentiment Analysis**")
		st.write("The histograms shows the distribution of sentiment values in the raw and cleaned data. The black dashed line represents the average sentiment value, while the orange dashed line represents the median. The sentiment values range from negative to positive, with the majority falling around the mean and median.")
		st.write("**Clean Data Sentiment Analysis**")
		st.write("After cleaning the data, the sentiment analysis was performed again. The histogram illustrates the distribution of sentiment values in the cleaned data. Similar to the raw data, the average sentiment value is represented by the black dashed line, and the median by the orange dashed line. Comparing with the raw data, cleaning seems to have affected the sentiment values, possibly shifting the distribution. Overall, we have:")
		st.write("- Positively skewed sentiments, mean is greater than median.")
		st.write("- Majority of the tweets are around neutral sentiment")
		st.write("- Cleaned data has a greater mean than raw data; however, the central tendency show a similar trend for both.")
	
		st.subheader("Compare most used hashtags for sentiment classes")
		st.image("Images//Hashtag News.png")
		st.write("For the News sentiment class, #climate, #climatechange, #environment and #news are the most popular hashtags.")
		st.image("Images//Hashtag Pro.png")
		st.write("For the Pro sentiments, hashtags like #beforetheflood, #environment, #climatechange, and #imvotingbecause can be extrapolated to indicate that the general context of the tweets is positive, with the tweets believing climate change is real")
		st.image("Images//Hashtag Anti.png")
		st.write("The most prevalent hashtags associated with sentiments = -1 appeared to be #fakenews, #climatescam, and #cnn. As a result, it is clear that the prevailing mood is negative. This is to be anticipated from tweets that deny man-made climate change.")
		st.image("Images//Hashtag Neutral.png")
		st.write("The tweets do not explicitly state whether they are for or against climate change, as one might expect given that they do not support or reject the belief in man-made climate change.")
	
	# Building out the "Home" page
	if selection == "Home":
		st.title("Welcome to Our Project")
		st.image("Images//Hand Planet.jpg")

		st.subheader("Brief Introduction")
		st.write("Welcome, reader, to our project. We, as Databotics, would like to introduce you to our app that contains various informative pages, as well as high end capabilities for the user. The purpose of our app is to provide a tool that summarises our project, briefly elaborates our collaboration with the Non-Governmental-Organisation(NGO) Friends of The Earth(FOTE) and what were have been trying to achieve, and to also spread the awareness of the negative impact caused by man made climate change.")

		st.image("Images//Earth Drop.jpg")
		st.subheader("Exploration")
		st.write("This app consists of several pages that are very informative and each page takes the user on the journey that was taken by the project team on the various tasks they completed. In the Predictions page we have our models that are able to identify the various categories of texts that have been put by the user. We also have an About Us page that introduces our company. We then have pages that begin with Project, and they discuss the team that was working on the tasks and the steps that were taken. Lastly, we have our Exploratory Data Analysis Page that contains the visuals and graphics of our project.")
	
		st.subheader("It's Time to Browse and Enjoy the App!!!!")


	
	# Building the Abous Us page
	if selection == "About Us":
		st.title("Databotics Solutions")
		st.subheader("Where Innovation Meets Excellence")
		st.image("Images//Databotics.png")
		
		st.subheader("Who We Are?")
		st.write("Databotics is a leading organization with rich heritage founded by visionaries from Nigeria and South Africa. We have established over 25 branches globally with our headquarters based in Cape Town, South Africa. We have given ourselves the purpose and destiny to provide unique and novel products that add value and make a difference amongst people and organisations worldwide. It is not only by our products that we are recognised, but by the solutions we offer for communities and organisations that we cooperate with.")
		st.subheader("Our History")
		st.write("To be where we are today, we owe it to our founders, Mr. Olufela Nwadike and Mrs Thandeka Nzimande, who were graduates at the institution called Explore AI and graduated as highly regarded Data Scientists. They founded the organisation in the year 2001, and have grown the company into a strong competitor within the industries of Artificial Intelligence, Machine Learning and Data Science. Originally the company was named Logic Boom, but it went through rebranding and various structural changes that resulted with the name Databotics in the year 2012. The name accommodates the direction and core of what the company has been trying to achieve all these years.")
		st.subheader("Our Mission")
		st.write("Our mission is to empower organizations with data-driven decision-making capabilities. We believe that every piece of data holds untapped potential, and by employing advanced analytics and cutting-edge technologies, we strive to unlock valuable insights that drive innovation, efficiency, and success for our clients.")
		st.subheader("Our Vision")
		st.write("We envision a future where data is seamlessly integrated into every aspect of business operations. We aim to be a major catalyst for positive change, where organizations harness the power of data to not only stay competitive but to lead and innovate in their respective industries.")
		st.subheader("Our Values")
		st.write("- Accuracy")
		st.write("- Innovation")
		st.write("- Integrity")
		st.write("- Service Excellence")
		st.write("- Continuous Learning")
		st.write("- Collaboration")
		st.write("- Client-Centric Approach.")

		st.subheader("What We Offer")
		st.write("- AI and ML ")
		st.write("- Data and Marketing Analytics")
		st.write("- Software Development")
		st.write("- Condulting Services")

	# Building the Project Summary page
	if selection == "Project: Summary":
		st.title("The Project")
		st.subheader("Brief Introduction to the Project")
		st.image("Images//Global.jpg")

		st.subheader("Project Overview")
		st.write("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
		st.write("With this context, Friends of the Earth is collaborating with Databotics Solutions to create a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.")
		
		st.subheader("Problem Statement")
		st.write("In order to mitigate the risk associated with climate change, our client is building toward lessening their environmental impact or carbon footprint by offering products and services that are environmentally friendly and sustainable, in line with their values and ideals. Our client will like to determine how people perceive climate change, and whether or not they believe it is a real threat. This will go a long way to add to their market research efforts in gauging how their product/service may be received.")
		st.write("Furthermore, we will be assisting them with their ad campaign that will assist in raising awareness of the effects of meat products and significance of moving to meat products that are plant based. In addition, with the development of our app, we will assist them with providing valuable information that aims to combat climate change and promotes plant-based meat products. This app will also contain information of the projection within which Friends of the Earth and Databotics collaborated in.")
		st.image("Images//Meat Plant.jpg")

		st.subheader("Project Objectives")
		st.write("Providing an accurate and robust solution to this task gives our client access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.")
		st.write("Databotics Solutions is task with creating a sentiment analysis model that will be able to classify whether or not a person believes in climate change, based on their novel tweet data.")
		st.write("In order to achieve the objectives, our company will do the following:")
		st.write("- Analyse the supplied data.")
		st.write("- Identify potential errors in the data and clean the existing data set.")
		st.write("- Determine if additional features can be added to enrich the data set.")
		st.write("- Build a model that is capable of predicting tweet.")
		st.write("- Evaluate the accuracy of the best machine learning model.")
		st.write("- Determine what features were most important in the model’s prediction decision, and")
		st.write("- Explain the inner working of the model to a non-technical audience.")

		st.subheader("Data Description and Sources")
		st.write("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes, which are described below.")
		st.write("Class Description:")
		st.write("- [2] News: the tweet links to factual news about climate change")
		st.write("- [1] Pro: the tweet supports the belief of man-made climate change")
		st.write("- [0] Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
		st.write("- [-1] Anti: the tweet does not believe in man-made climate change Variable definitions")
		st.write("Features:")
		st.write("- sentiment: Which class a tweet belongs in (refer to Class Description above)")
		st.write("- message: Tweet body")
		st.write("- tweetid: Twitter unique id")
		st.write("Files provided:")
		st.write("- train.csv: This data will be used to train our model.")
		st.write("- test.csv: This data will be used to test our model.")
		st.image("Images//Farming.jpg")

		st.subheader("Project Deliverables")
		st.write("Data Preprocessing:")
		st.write("- Clean and preprocess the dataset, handling missing values, and encoding categorical variables as necessary.")
		st.write("Exploratory Data Analysis(EDA):")
		st.write("- Conduct EDA to understand the relationships between weather features, energy generation, and the energy shortfall.")
		st.write("Feature Engineering:")
		st.write("- Create new features if necessary and select relevant features for modeling.")
		st.write("Model Building:")
		st.write("- Develop and train a regression model to predict the daily energy shortfall using the selected features.")
		st.write("Model Evaluation:")
		st.write("- Evaluate the model's performance using appropriate metrics such as mean squared error, mean absolute error, and R-squared.")
		st.write("Model Deployment:")
		st.write("- Deploy the trained model for making predictions on new data.")
		st.write("Documentation:")
		st.write("- Create documentation that explains the data, methodology, and model for future reference.")
		st.write("Project Timeline:")
		st.write("- The project will be conducted in phases, with each phase having specific tasks and deadlines. The timeline for the project will be determined by the project manager in consultation with the team members.")
		st.write("Data Volume")
		st.write("- The dataset contains nearly 15819 rows, and its size may impact data processing and model training times. Appropriate hardware and software resources will be allocated to handle this volume effectively.")
		st.write("Collaboration:")
		st.write("- The project team will collaborate closely to ensure the successful completion of each phase. Communication channels will be established to facilitate teamwork and knowledge sharing.")
	
	# Building out the "Project: Process Descriptions" page
	if selection == "Project: Process Descriptions":
		st.image("Images//Hands World.webp")
		st.title("Descriptions of Project Steps")
		st.write("In this page we will briefly elaborate on the core steps that the project team had undergone in order to achieve desired outcomes.")

		st.title("Data Pre-processing")
		st.image("Images//Preprocessing.png")
		st.subheader("Description:")
		st.write("Data Preprocessing includes the steps we need to follow to transform or encode data so that it may be easily parsed by the machine. The main agenda for a model to be accurate and precise in predictions is that the algorithm should be able to easily interpret the data's features.")
		st.subheader("Importance:")
		st.write("The majority of the real-world datasets for machine learning are highly susceptible to be missing, inconsistent, and noisy due to their heterogeneous origin.")
		st.write("Applying data mining algorithms on this noisy data would not give quality results as they would fail to identify patterns effectively. Data Processing is, therefore, important to improve the overall data quality.")
		st.write("Duplicate or missing values may give an incorrect view of the overall statistics of data. Outliers and inconsistent data points often tend to disturb the model’s overall learning, leading to false predictions. Quality decisions must be based on quality data. Data Preprocessing is important to get this quality data, without which it would just be a Garbage In, Garbage Out scenario.")

		st.title("Exploratory Data Analysis")
		st.image("Images//EDA.jpg")
		st.subheader("What is EDA?")
		st.write("Exploratory Data Analysis (EDA) is one of the techniques used for extracting vital features and trends used by machine learning and deep learning models in Data Science. Thus, EDA has become an important milestone for anyone working in data science. The Data Science field is now very important in the business world as it provides many opportunities to make vital business decisions by analyzing hugely gathered data. Understanding the data thoroughly needs its exploration from every aspect. The impactful features enable making meaningful and beneficial decisions; therefore, EDA occupies an invaluable place in Data science.")
		st.subheader("Objectives of EDA")
		st.write("- Identifying and removing data outliers")
		st.write("- Identifying trends in time and space")
		st.write("- Uncover patterns related to the target")
		st.write("- Creating hypotheses and testing them through experiments")
		st.write("- Identifying new sources of data")
		st.subheader("Types of EDA")
		st.write("There are three main types of EDA:")
		st.write("1. Univariate")
		st.write("2. Bivariate")
		st.write("3. Multivariate")
		st.subheader("Univariate Analysis")
		st.write("Examine Individual Variables: Analyze each variable independently to understand its distribution, central tendency, spread, and outliers. Visualize Distributions: Use histograms, box plots, and summary statistics to describe the data.")
		st.subheader("Bivariate and Multivariate Analysis")
		st.write("Explore Relationships: Investigate relationships between variables using scatter plots, correlation matrices, and cross-tabulations. Identify Patterns: Look for patterns, trends, or associations between different variables. Group Comparisons: Compare groups within categorical variables.")

		st.title("Feature Engineering")
		st.image("Images//F Engineering.jpg")
		st.subheader("Defining Feature Engineering")
		st.write("The feature engineering pipeline is the preprocessing steps that transform raw data into features that can be used in machine learning algorithms, such as predictive models. Feature engineering consists of creation, transformation, extraction, and selection of features, also known as variables, that are most conducive to creating an accurate ML algorithm. These processes entail:")
		st.subheader("Feature Creation")
		st.write("Creating features involves identifying the variables that will be most useful in the predictive model. This is a subjective process that requires human intervention and creativity. Existing features are mixed via addition, subtraction, multiplication, and ratio to create new derived features that have greater predictive power.")
		st.subheader("Transformations")
		st.write("Transformation involves manipulating the predictor variables to improve model performance; e.g. ensuring the model is flexible in the variety of data it can ingest; ensuring variables are on the same scale, making the model easier to understand; improving accuracy; and avoiding computational errors by ensuring all features are within an acceptable range for the model.")
		st.subheader("Feature Extraction")
		st.write("Feature extraction is the automatic creation of new variables by extracting them from raw data. The purpose of this step is to automatically reduce the volume of data into a more manageable set for modeling. Some feature extraction methods include cluster analysis, text analytics, edge detection algorithms, and principal components analysis.")
		st.subheader("Feature Selection")
		st.write("Feature selection algorithms essentially analyze, judge, and rank various features to determine which features are irrelevant and should be removed, which features are redundant and should be removed, and which features are most useful for the model and should be prioritized.")

		st.title("Modelling")
		st.image("Images//Modelling.webp")
		st.subheader("Description")
		st.write("Models are algorithms whose instructions are induced from a set of data and are then used to make predictions, recommendations, or prescribe an action based on a probabilistic assessment. The model uses algorithms to identify patterns in the data that form a relationship with an output. Models can predict things before they happen more accurately than humans, such as catastrophic weather events or who is at risk of imminent death in a hospital.")
		st.header("Key Steps")
		st.subheader("Choosing a Model")
		st.write("A machine learning model determines the output you get after running a machine learning algorithm on the collected data. It is important to choose a model which is relevant to the task at hand. Over the years, scientists and engineers developed various models suited for different tasks like speech recognition, image recognition, prediction, etc. Apart from this, you also have to see if your model is suited for numerical or categorical data and choose accordingly.")
		st.subheader("Training a Model")
		st.write("Training is the most important step in machine learning. In training, you pass the prepared data to your machine learning model to find patterns and make predictions. It results in the model learning from the data so that it can accomplish the task set. Over time, with training, the model gets better at predicting.")
	
		st.title("Model Performance")
		st.image("Images//Model Performance.png")
		st.header("Evaluating the Model")
		st.write("After training your model, you have to check to see how it’s performing. This is done by testing the performance of the model on previously unseen data. The unseen data used is the testing set that you split our data into earlier. If testing was done on the same data which is used for training, you will not get an accurate measure, as the model is already used to the data, and finds the same patterns in it, as it previously did. This will give you disproportionately high accuracy.")
		st.header("Evaluation Methods")
		st.subheader("Confusion matrix:")
		st.write("A confusion matrix is an error matrix. It is presented as a table in which the predicted class is compared with the actual class. Understanding confusion matrices is of paramount importance for understanding classification metrics, such as recall and precision. The rows of a confusion matrix represent real values, while the columns represent predicted values.")
		st.write("Reading a confusion matrix is relatively simple:")
		st.write("- True Positive (TP): you predicted positive, the real value was positive")
		st.write("- True Negative (TN): you predicted negative, the real value was negative")
		st.write("- False Positive (FP): you predicted positive, the real value was negative")
		st.write("- False Negative (FN): you predicted negative, the real value was positive")
		st.subheader("Precision")
		st.write("Precision is defined as the ratio of True Positives count to total True Positive count made by the model. Precision = TP/(TP+FP) Precision can be generated easily using precision_score() function from sklearn library.")
		st.write("The function takes 2 required parameters:")
		st.write("1. Correct Target labels")
		st.write("2. Predicted Target labels")
		st.subheader("Recall")
		st.write("Recall is defined as the ratio of True Positives count to the total Actual Positive count. Recall = TP/(TP+FN) Recall is also called “True Positive Rate” or “sensitivity”.")
		st.write("Recall can be generated easily using recall_score() function from sklearn library. The function takes 2 required parameters:")
		st.write("1. Correct Target labels")
		st.write("2. Predicted Target labels")
		st.subheader("F1 Score")
		st.write("The F1 score is easily one of the most reliable ways to score how well a classification model performs. It is the weighted average of precision and recall, as defined by the equation below:")
		st.write("F1 = 2 [(Recall * Precision) / (Recall + Precision)]")
		st.subheader("Specificity (Selectivity, True Negative Rate)")
		st.write("Specificity is similar to sensitivity, only the focus is on the negative class. It is the proportion of true negative cases which were correctly identified as such. The equation for specificity is:")
		st.write("Specificity = (True Negative) / (True Negative + False Positive)")
		st.subheader("Fall-out (False Positive Rate)")
		st.write("Fall-out determines the probability of determining a positive value when there is no positive value. It is the proportion of actual negative cases that were incorrectly classified as positive. The equation for fall-out is:")
		st.write("Fall-out = (False Positive) / (True Negative + False Positive)")
		st.subheader("Miss Rate (False Negative Rate)")
		st.write("Miss rate can be defined as the proportion of positive values that were incorrectly classified as negative examples.")
		st.write("Miss Rate = (False negative) / (True positive + False negative)")

		st.title("Model Explanations")
		st.image("Images//Model Explain.jpg")
		st.subheader("In Brief")
		st.write("The purpose of this section is to discuss the evaluated machine models that were picked, trained and tested. It is important to compare their results and elaborate further on the areas and stages where they have not performed well, and how they can be improved. ")
		st.write("This elaborated section is also important for the stakeholders, as they can be informed of the project outcomes that have been delivered by each tested model, and discuss with the project team on what is the way forward")
	
	# Building out the 'Project Team' page
	if selection == "Project: Team":
		st.title("Project Team")
		st.image("Images//Teamwork.jpg")
		st.header("The Team")
		st.subheader("Introducing the Project Team:")
		st.write("- Hlawulekani")
		st.write("- Edidong")
		st.write("- Lesego")
		st.write("- Pricilla")
		st.write("- Fransisca")
		st.write("- Boitemogelo")
		st.header("Brief Introduction to Team Roles:")
		st.subheader("Hlawulekani")
		st.image("Images//Hlawu.jpg")
		st.write("Hlawulekani is a PHD holder in Software Engineering from the University of Cape Town. With her hunger and passion, she earned various certificates in Systems Development, Cloud Computation and Data Engineering. She is currently employed by Databotics as the Chief Data Officer at its headquarters. Due to her in-depth knowledge of Supervised Machine Learning, she has been given the role of Project Leader.")
		st.subheader("Edidong")
		st.image("Images//Edicool.jpg")
		st.write("Edidong is a graduate of the Federal University of Technology, in Nigeria, where he graduated as a Software Engineer and Information Systems Analyst. He further completed his Masters in Systems Management at the University of Nigeria. He currently works for SystemSpecs. For the purpose of this project, Databotics has outsourced him, as he is highly regarded in the African continent, to be the Project Manager.")
		st.subheader("Lesego")
		st.image("Images//LSG.jpg")
		st.write("Lesego is a Information Technology graduate from the University of Wits, and also a Data Engineering graduate from Explore AI. She is also in the process of furthering her studies, as she is currently doing her Masters in Machine Learning and Cloud Computation. She is employed by Explore as a Data Analyst, and is highly regarded by the company. For this project, she will take on the role of Data Analyst.")
		st.subheader("Pricilla")
		st.image("Images//Pricilla.jpg")
		st.write("Pricilla is a graduate in Computer Science from the University of Pretoria. She is also an Explore AI graduate who completed her Data Science Course and is also enrolled in Data Engineering with the same academy. Her scholarship programme gave her the opportunity to be one of the best performing students, and was then employed by Databotics. For the purpose of this project she has been given the role of Data Scientist.")
		st.subheader("Fransisca")
		st.image("Images//Fra.jpeg")
		st.write("Fransisca is a Masters holder in Computer Science from the University of Lagos. She has also earned certificates in Data Science and Machine Learning. Databotics saw great potential in her during her time at Explore AI and decided to employ her. Today, she is the company’s Machine Learning Specialist, and has thus been given that role for the project as well.")
		st.subheader("Boitemogelo")
		st.image("Images//Temo.jpg")
		st.write("Boitemogelo is a Electrical Engineering Graduate from the University of Johannesburg. He has also adopted certificates in Big Data and Data Science with Explore AI. His hard work and grit caught the eye of Databotics, and has been employed by them as an Information Technology Manager. For the purpose of this project, he will also take on the role of IT and Resource Management.")



	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		
		modelChoice = st.radio("Choose a model", ("Logistic Regression","Linear SVC",))   
	
		if modelChoice == 'Logistic Regression':
			vect_text = tweet_cv.transform([tweet_text])
			#load pkl file with model and make predictions
			predictor = joblib.load(open(os.path.join("resources/logreg_model.pickle"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction == -1:
				st.success('Non-belief in man made climate change')
			elif prediction == 0:
				st.success('Neutral about climate change')
			elif prediction == 1:
				st.success('Belief in man made climate change')
			elif prediction == 2:
				st.success('Factual/news about climate change')
			else:
				st.success('Text has no classification')
			st.success("Text Classified as:{}".format(prediction))


		if modelChoice == 'Linear SVC':
			vect_text = tweet_cv.transform([tweet_text])
			#load pkl file with model and make predictions
			predictor = joblib.load(open(os.path.join("resources/lin_svc_model.pickle"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction == -1:
				st.success('Non-belief in man made climate change')
			elif prediction == 0:
				st.success('Neutral about climate change')
			elif prediction == 1:
				st.success('Belief in man made climate change')
			elif prediction == 2:
				st.success('Factual/news about climate change')
			else:
				st.success('Text has no classification')
			st.success("Text Classified as:{}".format(prediction))
	
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
