import streamlit as st
import preprocessor
import helper
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
# Customize Streamlit sidebar
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader(
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

        # Create a Streamlit container for the sidebar analysis controls
        sidebar_container = st.sidebar.container()

        # Customize the appearance of the analysis button
        with sidebar_container:
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

            # Activity Map
            st.title("Activity Map")

            col1, col2 = st.columns(2)

            with col1:
                most_active_day = helper.week_activity_map(selected_user, df)

                fig = px.bar(
                    most_active_day,
                    x=most_active_day.index,
                    y=most_active_day.values,
                    title="Most Active Day",
                    template = "seaborn",
                )

                fig.update_xaxes(title="Day of Week")
                fig.update_yaxes(title="Message Count")
                st.plotly_chart(fig,use_container_width=True, height = 200)

            with col2:
                most_active_month = helper.month_activity_map(selected_user, df)

                fig = px.bar(
                    most_active_month,
                    x=most_active_month.index,
                    y=most_active_month.values,
                    title="Most Active Month",
                    template = "seaborn",
                )

                fig.update_xaxes(title="Month")
                fig.update_yaxes(title="Message Count")
                st.plotly_chart(fig,use_container_width=True, height = 200)

            # Monthly Timeline
            st.title("Monthly Timeline")

            timeline = helper.monthly_timeline(selected_user, df)

            fig = px.line(
                timeline,
                x="time",
                y="message",
                title="Monthly Message Count",
                labels={"time": "Month-Year", "message": "Messages"},
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
                        template = "seaborn",
                    )

                    fig.update_xaxes(title="User")
                    fig.update_yaxes(title="Message Count")
                    st.plotly_chart(fig,use_container_width=True, height = 200)

                with col2:
                    st.dataframe(new_df.style.background_gradient(cmap="Blues"),height=500 ,width=1300,use_container_width=True)
                    dn=helper.fetch_most_active_user(df)
                    csv_data = new_df.to_csv(index=False)
                    st.download_button("Download Data", data=csv_data, file_name="user.csv", mime="text/csv", help='Click here to download the data as a CSV file', use_container_width=False)


            # Most Common Words
            st.markdown("## Most Common Words")

            most_common_df = helper.most_common_words(selected_user, df)

            fig = px.bar(
                most_common_df,
                x=1,  # Frequency column
                y=0,  # Word column
                title="Most Common Words",
                orientation="h",
                labels={1: "Frequency", 0: "Word"},  # Corrected column names
                template = "seaborn",
            )

            fig.update_xaxes(title="Frequency")
            fig.update_yaxes(title="Word")
            st.plotly_chart(fig,use_container_width=True, height = 200)


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
          
            custom_sort_order = [
                '00-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8',
                '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16',
                '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-00'
            ]

            # Define custom sorting orders
            custom_day_order = [
                'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
            ]

            # Group by 'day_name' and 'period' and count messages
            grouped_df = df.groupby(['day_name', 'period']).size().reset_index(name='message_count')
            st.dataframe(grouped_df)
            cv_data = grouped_df.to_csv(index=False)
            st.download_button("Download", data=cv_data, file_name="user.csv", mime="text/csv", help='Click here to download the data as a CSV file', use_container_width=False)
            # Create a scatter plot using Plotly Express
            scatter_fig = px.scatter(
                grouped_df, x='period', y='day_name', size='message_count',
                title="Weekly Activity Scatter Plot",
                color_discrete_sequence=['blue'],
                opacity=0.7,
            )

            # Customize the scatter plot layout and axis labels
            scatter_fig.update_layout(
                titlefont=dict(size=20),
                xaxis=dict(title="Time of Day", titlefont=dict(size=9), tickvals=custom_sort_order, ticktext=custom_sort_order),
                yaxis=dict(title="Day of Week", titlefont=dict(size=9), tickvals=custom_day_order, ticktext=custom_day_order)
            )

            # Display the scatter plot in Streamlit
            st.plotly_chart(scatter_fig, use_container_width=True)
            import seaborn as sns
            import matplotlib.pyplot as plt
            import numpy as np
            # # Pivot the grouped_df DataFrame for the heatmap 
            # # Group by 'day_name' and 'period' and count messages


            # # Pivot the data to create a matrix for the heatmap

            # st.dataframe(heatmap_data.values)
            # # Create the heatmap using Seaborn
            # plt.figure(figsize=(12, 6))
            # sns.set(font_scale=1.2)  # Adjust the font size if needed
            # sns.heatmap(heatmap_data.values, annot=False, cmap='Blues', cbar=True)

            # # Customize the heatmap's appearance
            # plt.title("Weekly Activity Heatmap")
            # plt.xlabel("Time of Day")
            # plt.ylabel("Day of Week")
            # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

            # # Display the heatmap in Streamlit
            # st.pyplot(plt)
            grouped_df = df.groupby(['day_name', 'period']).size().reset_index(name='message_count')
            heatmap_data = grouped_df.pivot_table(index='day_name', columns='period', values='message_count', aggfunc='sum').fillna(0)
            custom_sort_order = [
                '00-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8',
                '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16',
                '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-00'
            ]
            custom_day_order = [
                'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
            ]
            # heatmap_data = pd.DataFrame(np.random.randint(0, 100, size=(len(custom_day_order), len(custom_sort_order))),
            #                         columns=custom_sort_order, index=custom_day_order)

            # Create a Plotly heatmap figure
            fig = px.imshow(heatmap_data.values)

            # Customize the heatmap's appearance
            fig.update_layout(title="Weekly Activity Heatmap",
                            xaxis_title="Time of Day",
                            yaxis_title="Day of Week")

            # Display the heatmap in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            grouped_df = df.groupby(['day_name', 'period']).size().reset_index(name='message_count')
            heatmap_data = grouped_df.pivot_table(index='day_name', columns='period', values='message_count').fillna(0)

            # Define custom sorting orders
            custom_sort_order = [
                '00-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8',
                '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16',
                '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-00'
            ]
            custom_day_order = [
                'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
            ]

            # Create a Plotly heatmap figure
            fig = px.imshow(heatmap_data)

            # Customize the heatmap's appearance
            fig.update_layout(
                title="Weekly Activity Heatmap",
                xaxis_title="Time of Day",
                yaxis_title="Day of Week",
                xaxis=dict(tickvals=np.arange(len(custom_sort_order)), ticktext=custom_sort_order),
                yaxis=dict(tickvals=np.arange(len(custom_day_order)), ticktext=custom_day_order)
)

# Display the heatmap in Streamlit
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
