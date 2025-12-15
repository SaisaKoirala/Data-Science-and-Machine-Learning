# -----------Suspicious transaction detection helps find possible fraud in financial systems---------- 
# Here, we use the transaction and customer data, joining them through customer ID, and apply feature
# engineering and data preprocessing to make the data ready for analysis. By looking at
# location, age group, time of day, and transaction type, we can see which transactions are
# more likely to be suspicious. This helps in deciding where to focus monitoring and improve
# security.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

customer = pd.read_csv("Customer_Master.csv")
transaction = pd.read_csv("transactions.csv")

# displays top 4 data of customer
customer.head(4)

# displays top 4 data of transaction
transaction.head(4)

# Common columns that exist in both the customer and transaction datasets
duplicate_cols = set(customer.columns).intersection(set(transaction.columns))
duplicate_cols

duplicate_cols = duplicate_cols - {'customer_id'}
duplicate_cols

# Drop the duplicate columns from the transaction dataset
transaction = transaction.drop(columns= duplicate_cols)

#List common columns
duplicate_cols = set(customer.columns).intersection(set(transaction.columns))
duplicate_cols

dataset = customer.merge(transaction, on="customer_id", how="inner")

dataset.columns

dataset.isna().sum()

dataset.shape

dataset.drop(columns=["device", "os", "browser", "attempt_sequence"], inplace= True)

dataset.duplicated().sum()

dataset.isna().sum()

# Fill null values in device, OS, and browser columns with their respective mode
dataset["primary_device"].fillna(dataset["primary_device"].mode()[0], inplace= True)
dataset["primary_os"].fillna(dataset["primary_os"].mode()[0], inplace= True)
dataset["primary_browser"].fillna(dataset["primary_browser"].mode()[0], inplace= True)

#1. Checking transaction type for suspicious
sus_by_transaction_type = (
    dataset.groupby('transaction_type')['is_suspicious']
    .agg(['sum', 'count', 'mean'])
    .sort_values(by='mean', ascending=False)
)
print(sus_by_transaction_type)

# Reset index so transaction_type becomes a column again
sus_by_transaction_type = sus_by_transaction_type.reset_index()

# Select top 6
top6 = sus_by_transaction_type.sort_values('sum', ascending=False).head(6)

# Visualizaion
plt.figure(figsize=(8, 4))
plt.pie(
    top6['sum'],
    labels=top6['transaction_type'],
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title('Top 6 Transaction Types by Sum')
plt.tight_layout()
plt.show()

#2. Checking time of day for suspicious
sus_by_time = (
 dataset.groupby('time_of_day')['is_suspicious']
 .agg(['sum', 'mean'])
 .sort_values(by = 'sum', ascending=False)
 .reset_index()
)
print(sus_by_time)

#Visualization
plt.figure(figsize=(8,4))
plt.pie(
    sus_by_time['sum'],
    labels=sus_by_time['time_of_day'],
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title('Fraud Count by Time of Day')
plt.tight_layout()
plt.show()

#3. Checking location for suspicious
sus_by_location = (
 dataset.groupby('location')['is_suspicious']
 .agg(['sum', 'mean'])
 .sort_values(by = 'sum', ascending= False)
)
print(sus_by_location)

#Visualization
# Reset index so 'location' becomes a column
sus_by_location_reset = sus_by_location.reset_index()

# Select top 10 locations by fraud count (sum)
top10_locations = sus_by_location_reset.head(8)

plt.figure(figsize=(8, 4))
plt.pie(
    top10_locations['sum'],
    labels=top10_locations['location'],
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title('Top 8 Locations by Suspicious Transactions')
plt.tight_layout()
plt.show()

#4. Checking age group for suspicious
dataset['age_group'] = pd.Categorical(
 dataset['age_group'],
 ordered=True,
 categories=['18-25', '26-35', '36-45', '46-55', '56+']
)
dataset.groupby('age_group')['is_suspicious'].sum()

age_sus = dataset.groupby('age_group', as_index=False)['is_suspicious'].sum()

#Visualization
plt.figure(figsize=(8, 4))
plt.scatter(age_sus['age_group'], age_sus['is_suspicious'], s=200, color='#1f0c00')
plt.title("Fraud Cases by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Suspicious Count")
plt.show()

#5. Checking auth_method for suspicious 
sus_by_auth_method = (
    dataset.groupby('auth_method')['is_suspicious']
    .agg(['sum', 'count'])
    .sort_values(by='count', ascending=False)
    .reset_index()      # FIX
)
print(sus_by_auth_method)

#Visualization
plt.figure(figsize=(8,4))
plt.pie(
    sus_by_auth_method['count'],
    labels=sus_by_auth_method['auth_method'],   
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title('Fraud by Auth Method')
plt.tight_layout()
plt.show()

#Normalization
from sklearn.preprocessing import MinMaxScaler
ms_avg_monthly_income = MinMaxScaler(feature_range=(0, 1))
ms_amount = MinMaxScaler(feature_range=(0, 1))
ms_credit_score = MinMaxScaler(feature_range=(0, 1))
ms_avg_transaction_amount = MinMaxScaler(feature_range=(0, 1))
ms_amount_deviation = MinMaxScaler(feature_range=(0, 1))

#Initialize separate LabelEncoder instances for each categorical column
from sklearn.preprocessing import LabelEncoder
le_age_group = LabelEncoder()
le_home_location = LabelEncoder()
le_account_type = LabelEncoder()
le_mobile_banking_user = LabelEncoder()
le_primary_device = LabelEncoder()
le_primary_os = LabelEncoder()
le_primary_browser = LabelEncoder()
le_employment_status = LabelEncoder()
le_preferred_transaction_types = LabelEncoder()
le_location = LabelEncoder()
le_time_of_day = LabelEncoder()
le_status = LabelEncoder()
le_auth_method = LabelEncoder()
le_is_suspicious = LabelEncoder()
le_transaction_type = LabelEncoder()

# Apply label encoding to all categorical features
dataset["age_group"] = le_age_group.fit_transform(dataset["age_group"])
dataset["home_location"] = le_home_location.fit_transform(dataset["home_location"])
dataset["account_type"] = le_account_type.fit_transform(dataset["account_type"])
dataset["mobile_banking_user"] = le_mobile_banking_user.fit_transform(dataset["mobile_banking_user"])
dataset["primary_device"] = le_primary_device.fit_transform(dataset["primary_device"])
dataset["primary_os"] = le_primary_os.fit_transform(dataset["primary_os"])
dataset["primary_browser"] = le_primary_browser.fit_transform(dataset["primary_browser"])
dataset["employment_status"] = le_employment_status.fit_transform(dataset["employment_status"])
dataset["preferred_transaction_types"] = le_preferred_transaction_types.fit_transform(dataset["preferred_transaction_types"])
dataset["location"] = le_location.fit_transform(dataset["location"])
dataset["time_of_day"] = le_time_of_day.fit_transform(dataset["time_of_day"])
dataset["status"] = le_status.fit_transform(dataset["status"])
dataset["auth_method"] = le_auth_method.fit_transform(dataset["auth_method"])
dataset["is_suspicious"] = le_is_suspicious.fit_transform(dataset["is_suspicious"])
dataset["transaction_type"] = le_transaction_type.fit_transform(dataset["transaction_type"])

dataset.dtypes

dataset['transaction_id'].nunique()

dataset['transaction_date'].nunique()

dataset.drop(columns=['ip_address', 'transaction_id', 'transaction_date'], inplace=True)

X = dataset.drop(columns=['is_suspicious'])
y = dataset['is_suspicious']
X.ndim

# Split data into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train an SVM classifier and make predictions on the test set
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model's accuracy on the test set
model.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot heatmap
plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - SVM (RBF Kernel)")
plt.show()










