# Bank-Loan-Modelling
<b>Project Title: Marketing Campaign for Banking Products</b>

<b>This project is a part of Internship Studio Machine Learning (Training + Internship)</b>

<b>Data Description:</b> <br>
The file Bank.xls contains data on 5000 customers. The data include customer
demographic information (age, income, etc.), the customer's relationship with the bank
(mortgage, securities account, etc.), and the customer response to the last personal
loan campaign (Personal Loan).

Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was
offered to them in the earlier campaign.

<b>Data:</b> https://www.kaggle.com/itsmesunil/bank-loan-modelling/download

<b>Context:</b><br>
The bank has a growing customer base. The bank wants to increase borrowers (asset
customers) base to bring in more loan business and earn more through the interest on
loans. So , the bank wants to convert the liability based customers to personal loan
customers. (while retaining them as depositors). A campaign that the bank ran last year
for liability customers showed a healthy conversion rate of over 9% success. The
department wants you to build a model that will help them identify the potential
customers who have a higher probability of purchasing the loan. This will increase the
success ratio while at the same time reduce the cost of the campaign.

<b>Attribute Information:</b><br>
● ID: Customer ID<br>
● Age: Customer's age in completed years<br>
● Experience: #years of professional experience<br>
● Income: Annual income of the customer ($000)<br>
● ZIP Code: Home Address ZIP code.<br>
● Family: Family size of the customer<br>
● CCAvg: Avg. spending on credit cards per month ($000)<br>
● Education: Education Level. 1: Undergrad; 2: Graduate; 3:
Advanced/Professional<br>
● Mortgage: Value of house mortgage if any. ($000)<br>
● Personal Loan: Did this customer accept the personal loan offered in the last
campaign?<br>
● Securities Account: Does the customer have a securities account with the bank?<br>
● CD Account: Does the customer have a certificate of deposit (CD) account with
the bank?<br>
● Online: Does the customer use internet banking facilities?<br>
● Credit card: Does the customer use a credit card issued by the bank?<br>

<b>Objective:</b>
The classification goal is to predict the likelihood of a liability customer buying personal
loans.

<b>Steps and tasks:</b>
1. Import the datasets and libraries, check datatype, statistical summary, shape, null
values etc
2. Check if you need to clean the data for any of the variables
3. EDA: Study the data distribution in each attribute and target variable, share your
findings.<br>
● Number of unique in each column?<br>
● Number of people with zero mortgage?<br>
● Number of people with zero credit card spending per month?<br>
● Value counts of all categorical columns.<br>
● Univariate and Bivariate analysis<br>
4. Apply necessary transformations for the feature variables
5. Normalise your data and split the data into training and test set in the ratio of 70:30
respectively
6. Use the Logistic Regression model to predict the likelihood of a customer buying
personal loans.
7. Print all the metrics related for evaluating the model performance
8. Build various other classification algorithms and compare their performance
9. Give a business understanding of your model


