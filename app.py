import streamlit as st
import preprocessor
import helper
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import advertools as adv
import plotly.graph_objects as go

# Download the VADER lexicon (run only once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Map VADER sentiment score to sentiment categories

def map_sentiment(score):
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score >= -0.1:
        return "Neutral"
    elif score >= -0.5:
        return "Negative"
    else:
        return "Very Negative"

# Set Streamlit page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="✅",
    layout="wide",
)
st.title(" :bar_chart: WhatsApp Chat Analyzer")

# Removed the sidebar code

# Uploaded file handling
uploaded_file = st.file_uploader(
    "Upload Your WhatsApp Group Exported (without Media) txt file", type="txt"
)

if uploaded_file is not None:
    try:
        # To read file as bytes:
        bytes_data = uploaded_file.read()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # Perform sentiment analysis using VADER and add the results to the DataFrame
        df['Sentiment Score'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])

        # Map sentiment scores to sentiment categories
        df['Sentiment'] = df['Sentiment Score'].apply(map_sentiment)

        with st.expander("View Data"):
            # Use width='100%' to make the table fill the full width
            st.dataframe(df.style.background_gradient(cmap="Blues"), height=500 ,width=1300,use_container_width=True)
            csv = df.to_csv(index=False)

            # Set use_container_width=False to make the button fill the full width
            st.download_button("Download Data", data=csv, file_name="Whatsapp.csv", mime="text/csv", help='Click here to download the data as a CSV file', use_container_width=False)

        # Fetch unique users
        user_list = df["user"].unique().tolist()
        if "group_notification" in user_list:
            user_list.remove("group_notification")
        user_list.sort()
        user_list.insert(0, "Overall")
        df = df[df['user'] != 'group_notification']

        st.write("")  # Placeholder to create space
        selected_user = st.selectbox(
            "Show analysis wrt", user_list, key="selected-user"
        )
        analysis_button = st.button(
            "Show Analysis",
            key="analysis-button",
            help="Click to analyze the data",
        )
        if analysis_button:
            # Perform analysis and visualization here
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(
                selected_user, df
            )
            st.title("Top Statistics")
            with st.expander("View Data"):
                # Add your analysis and Plotly visualizations here

                # Top Statistics

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Messages", num_messages)
                with col2:
                    st.metric("Total Words", words)
                with col3:
                    st.metric("Media Shared", num_media_messages)
                with col4:
                    st.metric("Links Shared", num_links)

            # Daily Timeline
            st.title("Daily Timeline")

            daily_timeline = helper.daily_timeline(selected_user, df)

            fig = px.line(
                daily_timeline,
                x="date",
                y="message",
                title="Daily Message Count",
                labels={"date": "Date", "message": "Messages"},
                height=500, width = 1000,template="gridon",
            )

            # fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig,use_container_width=True)

            st.title("Activity Map")

            col1, col2 = st.columns(2)

            with col1:
                most_active_day = helper.week_activity_map(selected_user, df)

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=most_active_day.values,
                    theta=most_active_day.index,
                    fill='toself',
                    name="Most Active Day",
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(most_active_day.values)],
                        ),
                    ),
                    showlegend=False,
                    title="Most Active Day",
                    template="plotly_dark",
                    width=600,  # Adjust the width here
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                most_active_month = helper.month_activity_map(selected_user, df)

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=most_active_month.values,
                    theta=most_active_month.index,
                    fill='toself',
                    name="Most Active Month",
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(most_active_month.values)],
                        ),
                    ),
                    showlegend=False,
                    title="Most Active Month",
                    template="plotly_dark",
                    width=600,  # Adjust the width here
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
            # Monthly Timeline
            st.title("Monthly Timeline")

            timeline = helper.monthly_timeline(selected_user, df)

            fig = px.line(
                timeline,
                x="time",
                y="message",
                title="Monthly Message Count",
                labels={"time": "Month-Year", "message": "Messages"},
                template = "gridon",
            )

            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig,use_container_width=True, height = 200)

            # Find the person who sent the most messages
            if selected_user == "Overall":
                st.title("Most Active Users")

                x, new_df = helper.most_busy_users(df)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        x,
                        x=x.index,
                        y=x.values,
                        title="Most Active Users",
                        template = "ggplot2",
                    )
                    fig.update_xaxes(title="User")
                    fig.update_yaxes(title="Message Count")
                    st.plotly_chart(fig,use_container_width=True, height = 200)

                with col2:
                    st.subheader("User wise messege usege")
                    st.dataframe(new_df.style.background_gradient(cmap="Blues"),height=500 ,width=1300,use_container_width=True)
                    dn=helper.fetch_most_active_user(df)
                    csv_data = new_df.to_csv(index=False)
                    st.download_button("Download Data", data=csv_data, file_name="user.csv", mime="text/csv", help='Click here to download the data as a CSV file', use_container_width=False)

            # Most Common Words
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("## Most Common Words")

                most_common_df = helper.most_common_words(selected_user, df)

                fig = px.bar(
                    most_common_df,
                    x=1,  # Frequency column
                    y=0,  # Word column
                    title="Most Common Words",
                    orientation="h",
                    labels={1: "Frequency", 0: "Word"},  # Corrected column names
                    template = "plotly_dark",
                )
                fig.update_xaxes(title="Frequency")
                fig.update_yaxes(title="Word")
                st.plotly_chart(fig,use_container_width=True, height = 200)

            with col2:
                word_cloud = helper.created_word_cloud(selected_user, df)
                st.write("")
                st.write("")
                st.subheader(" Most Words in Wordcloud")
                # Create a Matplotlib figure and axis
                fig, ax = plt.subplots()
                ax.imshow(word_cloud)
                ax.axis("off")  # Turn off the axis

                # Display the word cloud in the Streamlit app using st.pyplot
                st.pyplot(fig, use_container_width=False)

            st.write("")
            # Emoji Analysis
            st.title("Emoji Analysis")

            emoji_df = helper.emoji_helper(selected_user, df)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Emoji Analysis Data")
                st.dataframe(emoji_df.style.background_gradient(cmap="Blues"), height=500, width=1000, use_container_width=True)
                st.download_button("Download Data", data = csv, file_name = "Emoji.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

            with col2:
                st.subheader("Emoji Analysis Chart")
                fig = px.pie(
                    emoji_df.head(),
                    values='count',
                    names='emoji',
                    hole=0.5,
                    template = "seaborn",
                )
                st.plotly_chart(fig,use_container_width=True, height = 200)

            helper.weekly_usage_analysis(selected_user, df)

            # Call the sentiment_analysis function here
            helper.sentiment_analysis(selected_user, df)
    except Exception as e:
        st.error(f"Error: {str(e)}")
