import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Walmart Sales Analysis by Maulana Malik",
    page_icon="assets/icon.png"
)

df = pd.read_csv('/Data/Pemrograman/Python/Artificial Intelligence/Streamlit/Latihan/walmart_sales_analysis/dataset/walmart.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

st.title("Walmart Sales Analysis\nby Maulana Malik Shafarulloh")
st.markdown("""
    Simple project to analyze sales from Walmart company. <br>
    Dataset used : <a href="https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data">https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data</a>
""", unsafe_allow_html=True)

st.markdown("""
    The dataset contains four files, which is ***'features.csv'***, ***'stores.csv'***, ***'train.csv'***, and ***'test.csv'***. We only use three of them because ***'train.csv'*** and ***'test.csv'*** are similar.
""")

st.markdown("""
    After merging, cleaning, and modifying, we get a dataset that looks like this :
""")
st.dataframe(df)
st.text(df.shape)

st.header("Weekly Sales Analysis")
st.subheader("Weekly Sales Over Time")
fig, ax = plt.subplots()
df['Weekly_Sales'].resample('W').sum().plot(kind='line', ax=ax)
ax.set_ylabel("Weekly Sales Mean (Million)")
ax.set_ylim(0)
ax.set_yticklabels((ax.get_yticks()/1000000))
st.pyplot(fig)

st.markdown("""
    Weekly Sales seems goes extremely high at the end of the year.<br>
    To be sure, let's take a look at the trends each year.
""", unsafe_allow_html=True)

st.subheader("Weekly Sales Each Year")
fig, ax = plt.subplots()
(pd.pivot_table(df, values='Weekly_Sales', columns='Year', index='Week')).plot(ax=ax)
ax.set_ylabel("Weekly Sales Mean (Thousand)")
ax.set_ylim(0)
ax.set_yticklabels((ax.get_yticks()/1000))
st.pyplot(fig)

st.markdown("""
    We now can confirm that the trend is same each year. We should find out why this happened.<br>
    If we look at the columns in the dataset, we can see column ***'IsHoliday'***. Maybe this is the reason why Weekly Sales are so high. We can see when the Holidays happen in the dataset.
""", unsafe_allow_html=True)

st.write(df[df['IsHoliday']].index.unique())

st.markdown("""
    If we look at the source of the dataset, we can see the date above correspond to ***'Super Bowl'***, ***'Labor Day'***, ***'Thanksgiving'***, and ***'Christmas'***.<br>
    Let's take a look at the mean sales if it is a holiday or not:
""", unsafe_allow_html=True)

st.subheader("Average Weekly Sales If the Day is Holiday")
fig, ax = plt.subplots()
sns.barplot(data=df, x='IsHoliday', y='Weekly_Sales', ax=ax)
st.pyplot(fig)

st.markdown("""
    As we can see the mean Weekly Sales is higher on Holiday
    Now we will see at the mean sales for each holiday.
""", unsafe_allow_html=True)

st.subheader("Average Weekly Sales Super Bowl")
fig, ax = plt.subplots()
sns.barplot(data=df, x='Super_Bowl', y='Weekly_Sales', ax=ax)
st.pyplot(fig)

st.subheader("Average Weekly Sales Labor Day")
fig, ax = plt.subplots()
sns.barplot(data=df, x='Labor_Day', y='Weekly_Sales', ax=ax)
st.pyplot(fig)

st.subheader("Average Weekly Sales Thanksgiving")
fig, ax = plt.subplots()
sns.barplot(data=df, x='Thanksgiving', y='Weekly_Sales', ax=ax)
st.pyplot(fig)

st.subheader("Average Weekly Sales Christmas")
fig, ax = plt.subplots()
sns.barplot(data=df, x='Christmas', y='Weekly_Sales', ax=ax)
st.pyplot(fig)

st.markdown("""
    It seems the highest sales were contributed by ***'Thanksgiving'***.
    Now let's see sales each store:
""", unsafe_allow_html=True)

st.subheader("Average Weekly Sales Each Store")
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=df, x='Store', y='Weekly_Sales', order=df.groupby(['Store']).mean().sort_values('Weekly_Sales', ascending=False).index, ax=ax)
st.pyplot(fig)

st.markdown("""
    Since the ***'Store'*** number just a number to identify each store, it almost mean nothing. The only thing we can conclude is that the sales of each store are different, and the 3 highest sales are obtained by store ***'20'***, ***'4'***, and ***'14'***.<br>
    But we can still see why sales can be high in some stores and low in others. Store has two attributes, ***'Size'*** and ***'Type'***. First, let's check the relationship between store ***'Size'*** and sales.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween Store Size and Weekly Sales")
fig, ax = plt.subplots()
sns.lineplot(x=df.groupby('Size').mean().index, y=df.groupby('Size').mean()['Weekly_Sales'], ax=ax)
st.pyplot(fig)

st.markdown("""
    It can be seen that the bigger the store size, the bigger the sales.<br>
    Next let's check what the store ***'Type'*** is by plotting the correlation between store ***'Type'*** and ***'Size'****.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween Store Size and Store Type")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='Type', y='Size', ax=ax)
st.pyplot(fig)

st.markdown("""
    From the plot we can conclude that Walmart groups its stores by size. So ***'A'*** is the largest store, ***'B'*** is the medium store, and ***'C'*** is the small store. It also mean that the ***'Type'*** of store have positive correlations with ***'Weekly Sales'*** as does ***'Size'*** have.<br>
    Now let's see Weekly Sales each Department.
""", unsafe_allow_html=True)

st.subheader("Weekly Sales each Department")
fig, ax = plt.subplots(figsize=(20, 15))
sns.barplot(data=df, x='Dept', y='Weekly_Sales', order=df.groupby(['Dept']).mean().sort_values('Weekly_Sales', ascending=False).index, ax=ax)
st.pyplot(fig)

st.markdown("""
    Like the ***'Store'*** number, the ***'Dept'*** number just a number to identify each department. The only thing we can conclude is that the sales of each department are different, and the 3 highest sales are obtained by department ***'92'***, ***'95'***, and ***'38'***.<br>
    Now let's take a look to other columns.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween Temperature and Weekly Sales")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=(df['Temperature']-32)*(5/9), y='Weekly_Sales', hue='IsHoliday', ax=ax)
plt.xlabel("Temperature (Celcius)")
st.pyplot(fig)

st.markdown("""
    It doesn't seem to have any pattern. The only thing we can see is as the temperature drops below 0, Weekly Sales also decrease.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween Fuel Price and Weekly Sales")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Fuel_Price', y='Weekly_Sales', hue='IsHoliday', ax=ax)
st.pyplot(fig)

st.markdown("""
    Looks like it doesn't have any pattern. The only thing we can conclude is as the fuel price goes too high, Weekly Sales is decreasing.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween Unemployment and Weekly Sales")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Unemployment', y='Weekly_Sales', hue='IsHoliday', ax=ax)
st.pyplot(fig)

st.markdown("""
    It also doesn't seem to have any pattern. The only thing we can see is as the unemployment amount goes too high, Weekly Sales is decreasing.
""", unsafe_allow_html=True)

st.subheader("Relationship Beetween CPI and Weekly Sales")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='CPI', y='Weekly_Sales', hue='IsHoliday', ax=ax)
st.pyplot(fig)

st.markdown("""
    There are 3 groups, but we can see in all of them are have sales, despite the fact that the CPI is higher.
""", unsafe_allow_html=True)

st.header("Conclusion")
st.markdown("""
    Before we jump to conclusion, if you look at the source of the dataset, you may notice that i remove column MarkDown.  Markdown are anonymized data related to promotional markdowns that Walmart is running. Since it ist anonymized, i think it's not necesarry to include it here, because we can't get any insight anyway.<br>
    <br>
    My conclusion is that Holidays are the most influential attribute in sales, especially Thanksgiving. It is important to prepare for this event. On the other hand, other days may need adjustments so that the sales target can still be achieved.
""", unsafe_allow_html=True)
