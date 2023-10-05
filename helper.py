# Import necessary libraries at the beginning of your helper.py
# import emoji
import pandas as pd
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import advertools as adv
import streamlit as st
import plotly.express as px

extract = URLExtract()

def fetch_stats(selected_data, data):
    if selected_data != "Overall":
        data = data[data["user"] == selected_data]

    num_messages = data.shape[0]
    num_media_messages = data[data['message'] == '<Media omitted>\n'].shape[0]

    link = []
    for message in data["message"]:
        link.extend(extract.find_urls(message))

    words = []
    for message in data["message"]:
        words.extend(message.split())

    return num_messages, len(words), num_media_messages, len(link)


def fetch_most_active_user(data):
    temp = data[data['user'] != 'group_notification']
    x = temp["user"].value_counts().head()
    result = round((temp["user"].value_counts() / temp.shape[0]) * 100, 1).reset_index().rename(
        {"index": "user", "user": "percentage"})
    return result


def created_word_cloud(selected_user, data):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    temp = data[data['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=800, height=250, min_font_size=10, background_color='white', colormap='RdYlGn',contour_color='#5d0f24',collocations=True)
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, data):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    temp = data[data['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_data = pd.DataFrame(Counter(words).most_common(20))
    return most_common_data

def emoji_helper(selected_user, data):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    emoji_summary = adv.extract_emoji(data['message'])['emoji_flat']
    df = pd.DataFrame(emoji_summary, columns=['emoji'])
    emoji_counts = df['emoji'].value_counts().reset_index()
    emoji_counts.columns = ['emoji', 'count']
    return emoji_counts


def monthly_timeline(selected_user, data):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    timeline = data.groupby(['year', 'month']).count()['message'].reset_index()

    # Combine 'month' and 'year' columns into a new 'time' column
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)

    return timeline


def daily_timeline(selected_user, data):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    daily_timeline = data.groupby('date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, data):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    return data['day_name'].value_counts()


def month_activity_map(selected_user, data):
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    return data['month'].value_counts()


def most_busy_users(df):
    temp = df[df['user'] != 'group_notification']
    x = temp['user'].value_counts().head()
    df = round((temp['user'].value_counts() / temp.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    df=df.rename(columns={'percent':'user','count':'percentage of usage'})
    return x,df

def weekly_usage_analysis(selected_user, df):
    st.title("Weekly Usage Analysis Chart")
    user_df=df
    # Filter the DataFrame for the selected user
    if selected_user != 'Overall':
        user_df = df[df['user'] == selected_user]

    grouped_df = user_df.groupby(['day_name', 'period']).size().reset_index(name='message_count')
    grouped_df['period'] = grouped_df['period'].str.replace(' ', '')  # Remove spaces
    heatmap_data = grouped_df.pivot_table(index='day_name', columns='period', values='message_count', aggfunc='sum').fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weekly Usage Chart")
        st.write("")
        st.dataframe(heatmap_data.style.background_gradient(cmap="Blues"), width=1000, use_container_width=False)
    with col2:
        fig = px.imshow(heatmap_data,color_continuous_scale="Blues")

        # Customize the heatmap's appearance
        fig.update_layout(title="Weekly Activity Heatmap",
                    xaxis_title="Time of Day",
                    yaxis_title="Day of Week",
                    width=1000)
         # Display the heatmap in Streamlit
        st.plotly_chart(fig, use_container_width=True)

def sentiment_analysis(selected_user, df):
    st.title("Sentiment Analysis")
    user_df=df
    # Filter the DataFrame for the selected user
    if selected_user != 'Overall':
        user_df = df[df['user'] == selected_user]

    sentiment_counts = user_df.groupby('Sentiment').size().reset_index(name='Counts')

    # Ensure 'date' column is in datetime format
    # Convert 'date' column to datetime format
    user_df['date'] = pd.to_datetime(user_df['date'], errors='coerce')

    # Create a new column 'month_year' by extracting the month and year
    user_df['month_year'] = user_df['date'].dt.to_period('M')

    # Group by 'month_year' and 'Sentiment' and count the occurrences
    sentiment_counts = user_df.groupby(['month_year', 'Sentiment']).size().reset_index(name='Counts')

    # Use a lambda function to format 'month_year' as a string
    sentiment_counts['month_year'] = sentiment_counts['month_year'].apply(lambda x: x.strftime('%b %Y'))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            names=sentiment_counts['Sentiment'],
            values=sentiment_counts['Counts'],
            title="Sentiment Distribution Monthly",
            hole=0.5,
            template="seaborn",
        )

        st.plotly_chart(fig, use_container_width=False)

    with col2:
        st.subheader("Message Sentiment Chart month-year wise")
        st.dataframe(sentiment_counts.style.background_gradient(cmap="Blues"), use_container_width=True)
    # Create a line chart for sentiment counts by formatted month and year
    
    fig = px.line(
        sentiment_counts,
        x='month_year',
        y='Counts',
        color='Sentiment',
        title="Sentiment Counts by Month and Year",
        labels={'Counts': 'Count'},
        template="seaborn",
    )

    st.plotly_chart(fig, use_container_width=True)