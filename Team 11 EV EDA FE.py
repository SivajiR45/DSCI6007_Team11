


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from mord import LogisticIT

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import neighbors 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import statsmodels.api as sm

from dmba import classificationSummary, gainsChart, liftChart
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import regressionSummary, exhaustive_search 
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import plotDecisionTree, classificationSummary, regressionSummary


from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#matplotlib inline
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Create data frame for data set.
ev_df = pd.read_csv('team11Dataset.csv')

# Display the first 10 records of heart_disease_df data frame.
print(ev_df.head(10))

# Determine dimensions of dataframe. 
print('Dimensions of dataframe:',ev_df.shape )
# It has 1025 rows and 14 columns.

ev_df.duplicated().sum()

ev_df.isna().sum()

# Display column data types in the dataframe
print('Datatypes of all the columns in the dataset')
print(ev_df.info())
ev_df.describe().T

# Display column data types in the dataframe before modification
print('Original Column data types')
print(ev_df.dtypes)

# Need to change all the variables with multiple classes to 'category'datatype 
ev_df.gender = ev_df.gender.astype('category')
ev_df.state = ev_df.state.astype('category')
ev_df.employment = ev_df.employment.astype('category')
ev_df.hsize = ev_df.hsize.astype('category')
ev_df.housit = ev_df.housit.astype('category')
ev_df.residence = ev_df.residence.astype('category')
#ev_df.zipcode = ev_df.zipcode.astype('category')
ev_df.buycar = ev_df.buycar.astype('category')
ev_df.home_evse = ev_df.home_evse.astype('category')
ev_df.work_evse = ev_df.work_evse.astype('category')
ev_df.town = ev_df.town.astype('category')
ev_df.highway = ev_df.highway.astype('category')
ev_df.home_parking = ev_df.home_parking.astype('category')
ev_df.work_parking = ev_df.work_parking.astype('category')
ev_df.RUCA = ev_df.RUCA.astype('category')
ev_df.Region = ev_df.Region.astype('category')
ev_df.Age_category = ev_df.Age_category.astype('category')
ev_df.education = ev_df.education.astype('category')
ev_df.hsincome = ev_df.hsincome.astype('category')
ev_df.range = ev_df.range.astype('category')
ev_df.bichoice = ev_df.bichoice.astype('category')

# Display category levels (attributes) and category type.
print(' ')
print('Category levels and changed variable type:')
print(ev_df.gender.cat.categories)
print(ev_df.gender.dtype)
print(ev_df.state.cat.categories)
print(ev_df.state.dtype)
print(ev_df.employment.cat.categories)
print(ev_df.employment.dtype)
print(ev_df.hsize.cat.categories)
print(ev_df.hsize.dtype)
print(ev_df.housit.cat.categories)
print(ev_df.housit.dtype)
print(ev_df.residence.cat.categories)
print(ev_df.residence.dtype)
print(ev_df.bichoice.cat.categories)
print(ev_df.bichoice.dtype)
# print(ev_df.zipcode.cat.categories)
# print(ev_df.zipcode.dtype)
print(ev_df.buycar.cat.categories)
print(ev_df.buycar.dtype)
print(ev_df.home_evse.cat.categories)
print(ev_df.home_evse.dtype)
print(ev_df.work_evse.cat.categories)
print(ev_df.work_evse.dtype)
print(ev_df.town.cat.categories)
print(ev_df.town.dtype)
print(ev_df.highway.cat.categories)
print(ev_df.highway.dtype)
print(ev_df.home_parking.cat.categories)
print(ev_df.home_parking.dtype)
print(ev_df.work_parking.cat.categories)
print(ev_df.work_parking.dtype)
print(ev_df.RUCA.cat.categories)
print(ev_df.RUCA.dtype)
print(ev_df.Region.cat.categories)
print(ev_df.Region.dtype)
print(ev_df.Age_category.cat.categories)
print(ev_df.Age_category.dtype)
print(ev_df.education.cat.categories)
print(ev_df.education.dtype)
print(ev_df.hsincome.cat.categories)
print(ev_df.hsincome.dtype)
print(ev_df.range.cat.categories)
print(ev_df.range.dtype)
print(ev_df.bichoice.cat.categories)
print(ev_df.bichoice.dtype)

ordinal_encoded_columns= ['state']

ordinal_encoder = OrdinalEncoder(categories='auto')
ordinal_encoded_data = ordinal_encoder.fit_transform(ev_df[ordinal_encoded_columns])

#Convert it to df
ordinal_encoded_data_df = pd.DataFrame(ordinal_encoded_data, index=ev_df.index,columns=['state'])
#ordinal_encoded_data_df.columns = ordinal_encoder.get_feature_names_out(input_features=ev_df[ordinal_encoded_columns])

#Extract only the columns that didnt need to be encoded
data_other_cols = ev_df.drop(columns=ordinal_encoded_columns)

#Concatenate the two dataframes : 
ev_df = pd.concat([ordinal_encoded_data_df, data_other_cols], axis=1)
print(ev_df)
ev_df.shape

# Display column data types in the dataframe after modification
print('Modified Column data types')
print(ev_df.dtypes)

ev_df.state = ev_df.state.astype('category')

print(ev_df.dtypes)

import plotly.graph_objs as go


# Load the data
data = pd.read_csv('team11Dataset.csv')

# Create a dictionary mapping state names to state codes
state_codes = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
               'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
               'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
               'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
               'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
               'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
               'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
               'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
               'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK','Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
               'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
               'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}

# Add a new column with the state codes
data['state_code'] = data['state'].map(state_codes)

state_data = data.groupby('state_code')['bichoice'].sum().reset_index()


# get state names and car interest counts as lists
state_names = state_data['state_code'].tolist()
car_interest_counts = state_data['bichoice'].tolist()

# create choropleth map using Plotly
fig = go.Figure(data=go.Choropleth(
    locations=state_names,
    z=car_interest_counts,
    locationmode='USA-states',
    colorscale='YlOrRd',
    colorbar_title="Number of Individuals<br>Interested to Buy Cars",
    marker_line_color='white',
    text=state_names
))

# set layout options
fig.update_layout(
    title_text='Individuals Interested to Buy EV Cars by State',
    geo=dict(
        scope='usa',
        projection=go.layout.geo.Projection(type='albers usa'),
        showlakes=False,
        lakecolor='rgb(255, 255, 255)'),
)

# show the map
fig.show()

#What Percentage Of people buy an EV ?
EV_count = ev_df['bichoice'].value_counts() 
EV_count

labels = ['Will buy an EV', 'Will not buy an EV']
slices = [3244, 2654]
explode = [0, 0.15]
# plotting in a pie chart
plt.pie(slices, labels =labels, explode = explode, shadow = True, startangle = 270, autopct='%1.1f%%')
plt.axis('equal')
plt.title("EV Buying Intention among the responents in the United states");

#updated
# percentage of buying intention
EV_Intention_count = ev_df['bichoice'].value_counts(normalize = True) * 100
EV_Intention_count

# plot the above count
EV_Intention_count.plot(kind = 'pie',autopct='%1.1f%%',explode = [0.1,0], startangle = 0)
plt.axis('equal')
plt.ylabel("")
plt.legend(title = 'Buying intention')
plt.title('What Percentage Of Individuals Buy an EV ?', fontdict= {'fontsize':14});

#updated
# percentage of each gender who have intention buy an EV
gender_per = ev_df.groupby('gender')['bichoice'].value_counts(normalize = True) 
gender_per

#Resetting the above groupby to a dataframe
gender_per = gender_per.reset_index(name = 'percentage')
gender_per

#Visualize the gender no show results
sns.catplot(data = gender_per, x = 'gender', y = 'percentage', hue = 'level_1', kind = 'bar');

#updated
# Function to create barplots that indicate percentage for each category.

def perc_on_bar(z):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(z) # length of the column
    plt.figure(figsize=(5,5))
    ax = sns.countplot(z,palette='Paired')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot
        
        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage 
    plt.show() # show the plot

perc_on_bar(ev_df['bichoice'])

perc_on_bar(ev_df['all_cars'])

perc_on_bar(ev_df['gender'])

perc_on_bar(ev_df['town'])

perc_on_bar(ev_df['highway'])

perc_on_bar(ev_df['education'])

perc_on_bar(ev_df['employment'])

perc_on_bar(ev_df['hsincome'])

perc_on_bar(ev_df['ev_cars'])

perc_on_bar(ev_df['home_parking'])

perc_on_bar(ev_df['buycar'])

plt.figure(figsize=(15,7))
sns.heatmap(ev_df.corr(),annot=True,vmin=-1,vmax=1,fmt='.2f',cmap='Spectral')
plt.show()

#updated
# Load the data into a pandas DataFrame
data = pd.read_csv('team11Dataset.csv')

# Group the data by residence area classification and calculate the average purchase decision for each group
grouped_data = data.groupby('RUCA')['bichoice'].value_counts(normalize = False) 
grouped_data = grouped_data.reset_index(name = 'count')
# Print the results
print(grouped_data)

sns.catplot(data = grouped_data, x = 'RUCA', y = 'count', hue = 'bichoice', kind = 'bar');
plt.box(on=None)
plt.tight_layout()
plt.title('Purchase decision by residence area classification', fontdict= {'fontsize':14})
plt.show()

#updated
# Load the data into a pandas DataFrame
data = pd.read_csv('team11Dataset.csv')

# Group the data by residence area classification and calculate the average purchase decision for each group
grouped_data = data.groupby('work_evse')['bichoice'].value_counts(normalize = False) 
grouped_data = grouped_data.reset_index(name = 'count')
# Print the results
print(grouped_data)

sns.catplot(data = grouped_data, x = 'work_evse', y = 'count', hue = 'bichoice', kind = 'bar');
plt.box(on=None)
plt.tight_layout()
plt.title('Purchase decision by Presence of Electric Vehicle Supply Equipment at work location', fontdict= {'fontsize':14})
plt.show()

#change axis name -rework 
# Load the data into a pandas DataFrame
data = pd.read_csv('team11Dataset.csv')

# Group the data by residence area classification and calculate the count purchase decision for each group
grouped_data1 = data.groupby('home_evse')['bichoice'].value_counts(normalize = False) 
grouped_data1 = grouped_data1.reset_index(name = 'count')
# Print the results
print(grouped_data1)

# Plot the results in a bar chart
sns.catplot(data = grouped_data1, x = 'home_evse', y = 'count', hue = 'bichoice', kind = 'bar');
plt.box(on=None)
plt.tight_layout()
plt.xlabel('Households with electrical outlet facility ')
plt.ylabel('purchase decision')
plt.title('Purchase decision by households who have an electrical outlet facility at their house parking')
plt.show()

#rework-pls updated y axis with number /counts
# Group the data by state and count the number of households expressing interest to purchase
grouped_data = data[data['bichoice'] == 1].groupby('state')['bichoice'].count()

# Sort the data by the count in descending order
sorted_data = grouped_data.sort_values(ascending=False)

# Print the top 10 states with the highest number of households expressing interest to purchase
print(sorted_data.head(10))

# Sort the data by the count in descending order
sorted_data = grouped_data.sort_values(ascending=False)

# Create a list of colors for each state
colors = plt.cm.get_cmap('Set3', len(sorted_data))

# Create a stacked bar plot of the data
plt.figure(figsize=(150,70))#change border size
for i, (state, count) in enumerate(sorted_data.items()):
    plt.bar(state, count, color=colors(i), width=0.8)
    plt.text(state, count + 100, str(count), ha='center', fontsize=60)
plt.xlabel('State', fontsize=100)
plt.xticks(rotation=90, fontsize=60)
plt.ylabel('Number of households expressing interest to purchase', fontsize=100)
plt.legend()
plt.title('Number of households expressing interest to purchase by state', fontsize=100)
plt.show()

#add labels to income range x axis -legend -rework
# Group the data by income range and count the number of households expressing interest to purchase
grouped_data = data.groupby('hsincome')['bichoice'].count()


plt.figure(figsize=(10, 6))

# Define the colors for the bars
colors = ['#8FB7C9', '#91A7D0', '#AC91D0', '#C590B3', '#D8A79D']

# Create the bar chart with the specified colors
plt.bar(grouped_data.index, grouped_data.values, color=colors, label='Number of households')

plt.xlabel('Annual Income Range')
plt.ylabel('Number of households expressing interest to purchase')
plt.title('Number of households expressing interest to purchase by annual income range')
plt.xticks(grouped_data.index, ['< $25k', '$25-50k', '$50-75k', '$75-100k', '> $100k'], rotation=45)

plt.legend()
plt.show()

data = pd.read_csv('team11Dataset.csv')
#rework change colour and enhance the look
# Filter the data to select only the rows where the interested_to_purchase column is 'Yes'
#data = data['bichoice']

# Filter the data to select only the rows where the employment column is '2' (student)
student_data = data[data['employment'] == 2]

# Count the number of households expressing interest to purchase for students and non-students
student_count = student_data['bichoice'].count()
non_student_count = data[data['employment'] != 2]['bichoice'].count()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(['Non-Student', 'Student'], [non_student_count, student_count], color=['#5DA5DA', '#FAA43A'])

# Set axis labels and title
plt.xlabel('Student Status', fontsize=14)
plt.ylabel('Number of Households Expressing Interest to Purchase', fontsize=14)
plt.title('Number of Households Expressing Interest to Purchase by Student Status', fontsize=16)

# Set font size for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set background color
plt.gca().set_facecolor('#F7F7F7')

# Show the plot
plt.show()

#add. colurs/labels,axis,legend,title,add box outsisde the chart- rework
# Load the dataset from a CSV file
df = pd.read_csv('team11Dataset.csv')

# Filter the dataset to only include rows where the home has off-street parking
df_with_parking = df[(df['home_parking'] == 1) | (df['home_parking'] == 2)]

# Calculate the percentage of households in each purchasing decision category
purchase_percentages = df_with_parking['bichoice'].value_counts(normalize=True) * 100

# Print the results
print(purchase_percentages)

# Set colors for the slices
colors = ['#5DA5DA', '#FAA43A']

# Create the pie chart
plt.pie(purchase_percentages, labels=['Yes', 'No'], colors=colors, autopct='%1.1f%%')

# Set the title and legend
plt.title('Purchase Decision for Households with Off-Street Parking', fontsize=14)
plt.legend(loc='upper right')

# Add a box outside the chart
#rect = plt.Rectangle((-0.015, -0.015), -1, -1, linewidth=1, edgecolor='gray', facecolor='none')
#plt.gca().add_patch(rect)

# Show the plot
plt.show()

#refer the question q1&q2 check the values displayed-rework -
# Load the dataset
dataset = pd.read_csv("team11Dataset.csv")

# Map the distance values to miles
#dataset['long_dist'] = dataset['long_dist'].map({1: 100, 2: 200, 3: 300, 4: 400})

# Calculate the percentage of households who made a purchase for each distance range
distance_purchase = dataset.groupby('long_dist')['bichoice'].sum()
# print(distance_purchase)
# print(dataset['long_dist'].unique())
distance_no_purchase = dataset.groupby('long_dist')['bichoice'].count() - distance_purchase
distance_purchase_percentage = distance_purchase / (distance_purchase + distance_no_purchase) * 100

# Print the results
print("Percentage of households who made a purchase for each distance range:")
print(distance_purchase_percentage)

# Plot the stacked bar plot
plt.bar(distance_purchase.index, distance_purchase, label="Purchase")
plt.bar(distance_no_purchase.index, distance_no_purchase, bottom=distance_purchase, label="No Purchase")
plt.xlabel("Distance (miles)")
plt.ylabel("Number of households")
plt.title("Effect of distance on purchasing decision")
plt.legend()
plt.show()

#change colurs and other stuffs- rework
# Group the households by their daily driving distance range and purchasing column
grouped_dataset = dataset.groupby([pd.cut(dataset['dmileage'], bins=[0, 10, 20, 30, 40, 50, 60]), 'bichoice'])

# Count the number of households in each group
grouped_counts = grouped_dataset.size()

# Calculate the percentage of households who made a purchase for each daily driving distance range
driving_distance_purchase = grouped_counts.unstack()[1] / (grouped_counts.unstack()[0] + grouped_counts.unstack()[1]) * 100

# Print the results
print(driving_distance_purchase)

#Add stacked bar chart

# Set the colors for the bars
colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0']

# Create the bar chart
plt.bar(driving_distance_purchase.index.astype(str), driving_distance_purchase.values, color=colors, width=0.4)

# Add x and y axis labels
plt.xlabel("Daily Average Driving Distance Range")
plt.ylabel("Percentage of Households Who Made a Purchase")

# Add a title
plt.title("Impact of Daily Average Driving Distance on Purchasing Decision")

# Add a horizontal grid
plt.grid(axis='y')

# Show the plot
plt.show()

# Group the households by their electric vehicle ownership and purchasing column
grouped_dataset = dataset.groupby(['ev_cars', 'bichoice'])

# Count the number of households in each group
grouped_counts = grouped_dataset.size()

# Calculate the percentage of households who made a purchase for each electric vehicle ownership status
electric_vehicle_purchase = grouped_counts.unstack()[1] / (grouped_counts.unstack()[0] + grouped_counts.unstack()[1]) * 100

#Rework-Change legends

# Define custom colors
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']

# Plot a bar chart of the results
electric_vehicle_purchase.plot(kind='bar', rot=0, color=colors)
plt.xlabel("No. of Electric Vehicles Owned", fontsize=8)
plt.ylabel("Percentage of Households Who Plan to Buy Another EV Car", fontsize=7)
plt.title("Impact of Electric Vehicle Ownership on Purchasing Decision", fontsize=8)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10, loc='upper left')
plt.show()

# Load the dataset
dataset = pd.read_csv("team11Dataset.csv")

# Subtract ev_cars from all_cars to get fuel_cars
dataset['Fuel_Cars'] = dataset['all_cars'] - dataset['ev_cars']

# Filter the households that currently own any number of fuel vehicles
fuel_vehicles_owned = dataset[dataset['Fuel_Cars'] > 0]

# Group the households by their interest in purchasing an EV as a second car
grouped_dataset = fuel_vehicles_owned.groupby('bichoice')

# Count the number of households in each group
grouped_counts = grouped_dataset.size()

# Calculate the percentage of households in each group
ev_interest = grouped_counts / grouped_counts.sum() * 100

# Print the results
print("Percentage of households interested in purchasing an EV as their next car:")
print(ev_interest)

# Plot the pie chart
ev_interest.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Percentage of households owning fuel cars interested in purchasing EV")
plt.legend(["No", "Yes"])
plt.show()

# Create a combined feature from 'home_parking' and 'home_evse'
dataset['home_charging_facility'] = (dataset['home_parking'] + dataset['home_evse']) / 2

# Display the first few rows of the dataset to verify the new feature
dataset[['home_parking', 'home_evse', 'home_charging_facility']].head()


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of the new combined feature 'work_charging_facility'
plt.figure(figsize=(10, 6))
sns.histplot(dataset['home_charging_facility'], bins=30, kde=True)
plt.title('Distribution of home Charging Facility Score')
plt.xlabel('home Charging Facility Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a combined feature from 'work_parking' and 'work_evse'
dataset['work_charging_facility'] = (dataset['work_parking'] + dataset['work_evse']) / 2

# Display the first few rows of the dataset to verify the new feature
dataset[['work_parking', 'work_evse', 'work_charging_facility']].head()


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of the new combined feature 'work_charging_facility'
plt.figure(figsize=(10, 6))
sns.histplot(dataset['work_charging_facility'], bins=30, kde=True)
plt.title('Distribution of Work Charging Facility Score')
plt.xlabel('Work Charging Facility Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(dataset['home_charging_facility'], bins=30, kde=True, color='blue', label='Home Charging Facility')
sns.histplot(dataset['work_charging_facility'], bins=30, kde=True, color='green', label='Work Charging Facility')
plt.title('Comparison of Home vs Work Charging Facility Scores')
plt.xlabel('Charging Facility Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def home():
    # Load the dataset (ensure the file path is correct and accessible)
    dataset = pd.read_csv('team11Dataset.csv')  # Replace with the correct path to your dataset

    # Group the households by their daily driving distance range and purchasing decision
    grouped_dataset = dataset.groupby([pd.cut(dataset['dmileage'], bins=[0, 10, 20, 30, 40, 50, 60]), 'bichoice']).size()
    driving_distance_purchase = grouped_dataset.unstack()[1] / (grouped_dataset.unstack()[0] + grouped_dataset.unstack()[1]) * 100

    # Set colors for the bar chart
    colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0']

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(driving_distance_purchase.index.astype(str), driving_distance_purchase.values, color=colors, width=0.4)
    plt.xlabel("Daily Average Driving Distance Range")
    plt.ylabel("Percentage of Households Who Made a Purchase")
    plt.title("Impact of Daily Average Driving Distance on Purchasing Decision")
    plt.grid(axis='y')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    # Embed the plot in the HTML template using base64 encoding
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
