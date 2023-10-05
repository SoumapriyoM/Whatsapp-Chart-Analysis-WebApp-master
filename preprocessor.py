import re
import pandas as pd

def get_time_slot(hour):
    if 0 <= hour < 4:
        return "Late Night"
    elif 4 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"
def preprocess(data):
    dissision_pattern = "\d{1,2}:\d{2}\s[AaPp][Mm]"
    dission = len(re.findall(dissision_pattern, data))

    if dission >= 3:
        pattern = "\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AaPp][Mm]\s-\s"
        pattern1 = "\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AaPp][Mm]"
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern1, data)

        df = pd.DataFrame({"user_message": messages, "date": dates})
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y, %I:%M %p")  # Updated format
        df = df[["date", "user_message"]]
    else:
        pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)

        df = pd.DataFrame({'user_message': messages, 'date': dates})
        # Convert message_date type
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y, %H:%M - ')  # Updated format

        df = df[["date", "user_message"]]

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # User name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df = df.drop(columns="date")
    
    # period = []
    # for hour in df[['day_name', 'hour']]['hour']:
    #     if hour == 23:
    #         period.append(str(hour) + ":" + str('00'))
    #     elif hour == 0:
    #         period.append(str('00') + ":" + str(hour + 1))
    #     else:
    #         period.append(str(hour) + ":" + str(hour + 1))
    df = df.rename(columns={'only_date': 'date'})
    df['period'] = df['hour'].apply(get_time_slot)
    df['period'] = df['period'].astype(str)  # Change the 'period' column to string data type with colons

    return df
