"""
Generate mock survey data for parkrun surveys.
Generates 500 rows for each survey CSV.
"""

import csv
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker('en_GB')

# ============================================================================
# UK Brand Survey Mock Data
# ============================================================================

def generate_brand_survey_row(respondent_id):
    """Generate a single row of mock brand survey data."""
    start_date = fake.date_time_this_year()
    end_date = start_date + timedelta(minutes=random.randint(3, 8))

    row = {
        'Respondent ID': respondent_id,
        'Collector ID': random.randint(1000000, 9999999),
        'Start Date': start_date.strftime('%m/%d/%Y %H:%M'),
        'End Date': end_date.strftime('%m/%d/%Y %H:%M'),
        'IP Address': f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
        'Email Address': fake.email(),
        'First Name': fake.first_name(),
        'Last Name': fake.last_name(),
        'Custom Data 1': '',
        'rq_flag': random.randint(1, 2),
        'Are you happy to consent to the above?': random.choice(['Yes', 'Yes', 'Yes', 'Yes', 'No']),  # Mostly yes
        'How did you first hear about parkrun?': random.choice([
            'Through word of mouth (e.g. friends, family, work colleagues etc.)',
            'Through media coverage (e.g. newspaper, TV, radio, podcast etc.)',
            'On social media (e.g. Facebook, Instagram etc.)',
            'Through an internet search',
            'Through a community group (e.g. running club, sports club, couch to 5k group etc.)',
            'From a healthcare professional (e.g. GP, nurse, physiotherapist etc.)',
            'Through a school',
            'Through a parkrun partner or sponsor',
            'I saw a parkrun event happening'
        ]),
        # Parkrun values - 5 statements, each with Strongly agree to N/A
        'Doing parkrun makes me feel healthier': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree', 'Disagree']),
        'Doing parkrun makes me feel happier': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree', 'Disagree']),
        'I feel more positive after I have participated at parkrun': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        'parkrun events are inclusive': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        'parkrun is a welcoming place for walkers': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        'parkrun is a welcoming place for people of all abilities': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        'parkrun feels like a community': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        # More parkrun values
        'parkrun is for people like me': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree', 'Disagree']),
        'parkrun values me as an individual': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree', 'Disagree']),
        'parkrun listens to my point of view': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree', 'Disagree']),
        'I feel welcome to participate at parkrun': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        'parkrun is a brand I trust': random.choice(['Strongly agree', 'Agree', 'Agree', 'Neither agree nor disagree']),
        # NPS question
        'How likely is it that you would recommend parkrun to a friend or colleague?': random.randint(0, 10),
        'What could we do to improve your experience of parkrun?': fake.paragraph(nb_sentences=2) if random.random() > 0.3 else '',
        # Communications frequency
        'Weekly Newsletter': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all', 'Don\'t know']),
        'parkrun UK or Global Facebook': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all', 'Don\'t know']),
        'A parkrun Event\'s Facebook': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        'parkrun UK or Global X / Twitter': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        'A parkrun Event\'s X / Twitter': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        'parkrun UK Instagram': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        'A parkrun Event\'s Instagram': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        'parkrun YouTube': random.choice(['Weekly or more often', 'Two or three times per month', 'About once per month', 'Every few months', 'Not at all']),
        # Sponsors
        'Can you name any current sponsors of parkrun UK?': random.choice(['Vitality', 'Brooks', 'Runna', 'Kenco', 'Multiple sponsors', '']),
        # Health insurance awareness
        'Aviva': random.choice(['Selected', '']),
        'Vitality': random.choice(['Selected', 'Selected', '']),
        'Bupa': random.choice(['Selected', '']),
        'The Exeter': random.choice(['Selected', '']),
        'WPA': random.choice(['Selected', '']),
        'AXA PPP': random.choice(['Selected', '']),
        'PruHealth': random.choice(['Selected', '']),
        'Zurich': random.choice(['Selected', '']),
        'AIG': random.choice(['Selected', '']),
        'None of these': random.choice(['Selected', '']),
        # Vitality questions
        'Vitality is currently a partner of parkrun. Were you aware of that before today?': random.choice(['Yes', 'No', 'Not sure']),
        # Vitality locations - multiple select
        'Vitality In the newsletter': random.choice(['Selected', '']),
        'Vitality In my results email': random.choice(['Selected', '']),
        'Vitality In other emails from parkrun': random.choice(['Selected', '']),
        'Vitality On social media (e.g. parkrun Facebook)': random.choice(['Selected', '']),
        'Vitality At parkrun events (e.g. on a flag or banner)': random.choice(['Selected', '']),
        'Vitality A visit from the sponsor at my event': random.choice(['Selected', '']),
        # Vitality usage
        'Have you taken out health or life insurance with Vitality?': random.choice(['Yes - I was a member prior to the partnership (before February 2017)', 'Yes - I have taken out membership since February 2017', 'No', 'Not sure']),
        'Was the policy you took out with Vitality personal (you contacted Vitality directly) or corporate (arranged through your employer)?': random.choice(['Personal', 'Corporate (through my workplace)', 'Not sure / prefer not to say']),
        'Did the partnership Vitality has with parkrun have an impact on your decision to take out the policy?': random.choice(['A major impact', 'Some impact', 'No impact at all', 'Don\'t know / prefer not to say']),
        'Would you consider using Vitality for your health and/or life insurance in future?': random.choice(['Yes', 'No', 'Not sure']),
        'Do you know more about who Vitality are and what they do as a result of the partnership with parkrun?': random.choice(['Yes', 'No', 'Not sure']),
        'Has your attitude towards Vitality changed as a result of the partnership with parkrun?': random.choice(['I feel more positive towards Vitality now', 'I feel less positive towards Vitality now', 'My attitude hasn\'t changed at all']),
        # Fitness brands
        'Coopah': random.choice(['Selected', '']),
        'Couch to 5k': random.choice(['Selected', '']),
        'Garmin Coach': random.choice(['Selected', '']),
        'Kiprun': random.choice(['Selected', '']),
        'Nike Run Club': random.choice(['Selected', '']),
        'Runna': random.choice(['Selected', 'Selected', '']),
        'Strava': random.choice(['Selected', '']),
        # Runna partnership
        'Runna is currently a partner of parkrun. Were you aware of that before today?': random.choice(['Yes', 'No', 'Not sure']),
        'Have you ever used the Runna running training app?': random.choice(['Yes - Prior to the partnership (before January 2025)', 'Yes - Only since January 2025', 'No', 'Not sure']),
        'Would you consider using the Runna running training app in the future?': random.choice(['Yes', 'No', 'Not sure']),
        'Do you know more about who Runna are and what they do as a result of the partnership with parkrun?': random.choice(['Yes', 'No', 'Not sure']),
        'Has your attitude towards Runna changed as a result of the partnership with parkrun?': random.choice(['I feel more positive towards Runna now', 'My attitude hasn\'t changed at all']),
        # Running shoes
        'Adidas': random.choice(['Selected', '']),
        'Asics': random.choice(['Selected', '']),
        'Brooks': random.choice(['Selected', 'Selected', '']),
        'Hoka One One': random.choice(['Selected', '']),
        'Inov-8': random.choice(['Selected', '']),
        'Mizuno': random.choice(['Selected', '']),
        'New Balance': random.choice(['Selected', '']),
        'Nike': random.choice(['Selected', '']),
        'On': random.choice(['Selected', '']),
        'Salomon': random.choice(['Selected', '']),
        'Saucony': random.choice(['Selected', '']),
        # Brooks partnership
        'Brooks is currently a partner of parkrun. Were you aware of that before today?': random.choice(['Yes', 'No', 'Not sure']),
        'Have you ever bought a pair of Brooks running shoes?': random.choice(['Yes - Prior to the partnership (before February 2020)', 'Yes - Only since February 2020', 'No', 'Not sure']),
        'Would you consider buying Brooks running shoes in future?': random.choice(['Yes', 'No', 'Not sure']),
        'Do you know more about who Brooks are and what they do as a result of the partnership with parkrun?': random.choice(['Yes', 'No', 'Not sure']),
        'Has your attitude towards Brooks changed as a result of the partnership with parkrun?': random.choice(['I feel more positive towards Brooks now', 'My attitude hasn\'t changed at all']),
        # Coffee brands
        'Café Direct': random.choice(['Selected', '']),
        'Costa': random.choice(['Selected', '']),
        'Costa Capsules': random.choice(['Selected', '']),
        'Douwe Egberts': random.choice(['Selected', '']),
        'Kenco': random.choice(['Selected', 'Selected', '']),
        'Kenco Millicano': random.choice(['Selected', '']),
        'L\'Or': random.choice(['Selected', '']),
        'Lavazza': random.choice(['Selected', '']),
        'Nescafe Azera': random.choice(['Selected', '']),
        'Nescafe Gold': random.choice(['Selected', '']),
        'Nescafe Original': random.choice(['Selected', '']),
        'Tassimo': random.choice(['Selected', '']),
        'Taylors': random.choice(['Selected', '']),
        # Kenco partnership
        'Kenco is currently a partner of parkrun. Were you aware of that before today?': random.choice(['Yes', 'No', 'Not sure']),
        'Have you ever purchased Kenco coffee?': random.choice(['Yes - Prior to the partnership (before March 2025)', 'Yes - Only since March 2025', 'No', 'Not sure']),
        'Would you consider buying Kenco coffee in future?': random.choice(['Yes', 'No', 'Not sure']),
        'Do you know more about who Kenco are and what they do as a result of the partnership with parkrun?': random.choice(['Yes', 'No', 'Not sure']),
        'Has your attitude towards Kenco changed as a result of the partnership with parkrun?': random.choice(['I feel more positive towards Kenco now', 'My attitude hasn\'t changed at all']),
        # Demographics
        'What is your ethnic group?': random.choice(['White', 'Asian', 'Black', 'Mixed', 'Other']),
        'Are your day-to-day activities limited because of a health condition or disability which has lasted, or is expected to last, at least 12 months?': random.choice(['Yes', 'No', 'Don\'t know / Prefer not to say']),
        'Athlete ID': random.randint(100000, 999999),
    }
    return row

# Generate brand survey data
print("Generating UK Brand Survey data (500 rows)...")
brand_survey_path = r'C:\Users\olive\OneDrive\Documents\GitHub\parkrun_survey_analysis\data\surveys\UK Brand Survey Blank Data - 15.5.26.csv'

# Get headers from the first generated row
first_row = generate_brand_survey_row(1)
headers = list(first_row.keys())

with open(brand_survey_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for i in range(1, 501):
        row = generate_brand_survey_row(i)
        writer.writerow(row)

print("Generated 500 rows for UK Brand Survey")

# ============================================================================
# UK Volunteer Experience Survey Mock Data
# ============================================================================

def generate_volunteer_survey_row(respondent_id):
    """Generate a single row of mock volunteer survey data."""
    start_date = fake.date_time_this_year()
    end_date = start_date + timedelta(minutes=random.randint(5, 12))

    row = {
        'Question': '',
        'Respondent ID': respondent_id,
        'Collector ID': random.randint(1000000, 9999999),
        'Start Date': start_date.strftime('%m/%d/%Y %H:%M'),
        'End Date': end_date.strftime('%m/%d/%Y %H:%M'),
        'IP Address': f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
        'Email Address': fake.email(),
        'First Name': fake.first_name(),
        'Last Name': fake.last_name(),
        'Custom Data 1': '',
        'Are you happy to consent to the above?': 'Yes',
        'When did you first volunteer at parkrun?': random.choice(['In the last 12 months', 'A year ago or longer', 'I don\'t remember / prefer not to say']),
        'Before I volunteered at parkrun, in the previous 3 years': random.choice([
            'I had no experience of volunteering in a sport or community context',
            'I had some experience of volunteering in a sport or community context',
            'I had lots of experience of volunteering in a sport or community context',
            'Don\'t know / prefer not to say'
        ]),
        'I found out about volunteering opportunities at parkrun through (please select one option only)': random.choice([
            'Friends', 'Family', 'A running club', 'The briefing before a parkrun event', 'The email appeal from a parkrun event',
            'Other volunteers at a parkrun event', 'parkrun newsletter', 'parkrun social media'
        ]),
        'The first time I volunteered, I made myself known to the event team by (please select one option only)': random.choice([
            'Speaking to a volunteer at the event and volunteering that day',
            'Speaking to a volunteer at the event and volunteering at a later date',
            'Writing my name on a physical/paper roster at the event',
            'Responding to a volunteer appeal email from the event team',
            'Emailing the event by replying to my results email',
            'Emailing the event independently',
            'Contacting the event team on social media (e.g. Facebook, Instagram)',
            'Being put forward by a friend or family member'
        ]),
        # Feelings before first volunteering
        'Excited to volunteer': random.choice(['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']),
        'Anxious about volunteering': random.choice(['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree', 'Disagree']),
        'Proud to volunteer': random.choice(['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']),
        # Feelings after first time
        'Positive': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'Inspired': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'Part of a team': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        # Motivations for starting
        'I was unable to walk/run e.g. injured': random.choice(['Selected', '']),
        'I didn\'t want to walk/run': random.choice(['Selected', '']),
        'To catch up with friends': random.choice(['Selected', 'Selected', '']),
        'I had a family member participating at parkrun': random.choice(['Selected', '']),
        'I had free time': random.choice(['Selected', 'Selected', '']),
        'It\'s fun': random.choice(['Selected', 'Selected', 'Selected', '']),
        'To do something meaningful (e.g give back to the community, help others)': random.choice(['Selected', 'Selected', 'Selected', '']),
        'To help make sure the event went ahead': random.choice(['Selected', 'Selected', '']),
        'To improve my mental health': random.choice(['Selected', '']),
        'To improve my physical health': random.choice(['Selected', '']),
        'To meet new people': random.choice(['Selected', 'Selected', '']),
        'To use/develop skills': random.choice(['Selected', '']),
        'To move closer to a parkrun volunteer milestone': random.choice(['Selected', '']),
        'As part of completing a parkrun 5k app challenge': random.choice(['Selected', '']),
        # Why didn't volunteer earlier
        'I may have volunteered earlier, but (please select one option only)': random.choice([
            'I didn\'t have time', 'It is difficult to commit to volunteering in advance', 'I wanted to run',
            'I volunteer elsewhere', 'I didn\'t feel confident enough', 'I didn\'t feel welcome to volunteer',
            'The event had enough volunteers already'
        ]),
        # Return to volunteer
        'Have you returned to volunteer again at parkrun, or volunteered once only?': random.choice([
            'I have returned and volunteered again at parkrun',
            'I have volunteered at parkrun once only',
            'Don\'t know / prefer not to say'
        ]),
        # Last time motivations (same options as first time)
        'I was unable to walk/run e.g. injured (last time)': random.choice(['Selected', '']),
        'To catch up with friends (last time)': random.choice(['Selected', 'Selected', '']),
        'It\'s fun (last time)': random.choice(['Selected', 'Selected', '']),
        'To do something meaningful (e.g give back to the community, help others) (last time)': random.choice(['Selected', 'Selected', '']),
        # Volunteer roles experience
        'Marshal - Easy / Difficult': random.choice(['Easy', 'Difficult']),
        'Marshal - Enjoyable / Not enjoyable': random.choice(['Enjoyable', 'Not enjoyable']),
        'Barcode scanning - Easy / Difficult': random.choice(['Easy', 'Difficult']),
        'Barcode scanning - Enjoyable / Not enjoyable': random.choice(['Enjoyable', 'Not enjoyable']),
        'Pre-event setup - Easy / Difficult': random.choice(['Easy', 'Difficult']),
        'Pre-event setup - Enjoyable / Not enjoyable': random.choice(['Enjoyable', 'Not enjoyable']),
        # Issues experienced
        'Have you experienced any issues whilst volunteering at parkrun?': random.choice(['Yes', 'No', 'No', 'No']),
        # Challenges and support
        'Dealing with disputes - Have experienced / Have not experienced': random.choice(['Have experienced', 'Have not experienced']),
        'Dealing with disputes - Comfortable dealing with / Not comfortable dealing with': random.choice(['Comfortable dealing with', 'Not comfortable dealing with']),
        'Getting enough volunteers - Have experienced / Have not experienced': random.choice(['Have experienced', 'Have experienced', 'Have not experienced']),
        'Getting enough volunteers - Comfortable dealing with / Not comfortable dealing with': random.choice(['Comfortable dealing with', 'Not comfortable dealing with']),
        # How easy to get volunteers
        'How easy or difficult do you find it to get enough volunteers at your event each week?': random.randint(0, 10),
        # Recognition importance
        'Words of appreciation from my event team': random.choice(['Extremely important', 'Very important', 'Moderately important', 'Slightly important']),
        'Words of appreciation from parkrunners': random.choice(['Extremely important', 'Very important', 'Moderately important', 'Slightly important']),
        'Volunteer credits': random.choice(['Extremely important', 'Very important', 'Moderately important', 'Slightly important']),
        'My volunteer milestone t-shirt': random.choice(['Extremely important', 'Very important', 'Moderately important', 'Slightly important']),
        # Experience on the day
        'I felt valued by the event team': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I felt valued by the parkrun community': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I got on well with my fellow volunteers': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'There was always someone I could go to for help': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'There was a positive team culture': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I volunteered because I wanted to': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        # Impact
        'My mental health has improved': random.choice(['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']),
        'My physical health has improved': random.choice(['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']),
        'I have learned new skills': random.choice(['Strongly agree', 'Agree', 'Strongly agree', 'Disagree']),
        'I feel happier': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I would recommend being a parkrun volunteer to others': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I intend to continue to volunteer at parkrun': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        # Skills developed
        'Communication & interpersonal': random.choice(['Selected', 'Selected', '']),
        'Leadership': random.choice(['Selected', '']),
        'Teamwork': random.choice(['Selected', 'Selected', 'Selected', '']),
        'Planning & organising': random.choice(['Selected', '']),
        # Organization beliefs
        'parkrun champions volunteering and the role of the volunteer': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        'I trust parkrun head office to make the right decisions': random.choice(['Strongly agree', 'Agree', 'Agree', 'Disagree']),
        # Event Director status
        'Are you an Event Director or Volunteer Ambassador?': random.choice(['Yes', 'No', 'No', 'No']),
        'What is your role at parkrun?': random.choice(['Event Director', 'Volunteer Ambassador', '']),
        # Recommendation
        'How likely is it that you would recommend volunteering at parkrun to a friend or colleague?': random.randint(6, 10),
        'My overall experience as a parkrun volunteer has been': random.choice(['Meaningful', 'Positive', 'Easy', 'Positive']),
        # Open feedback
        'What I like about volunteering at parkrun is...': fake.paragraph(nb_sentences=1),
        'If I could recommend one thing to improve my parkrun volunteer experience it would be...': fake.paragraph(nb_sentences=1),
        # Demographics
        'Do you have any physical or mental health conditions or illnesses lasting or expected to last 12 months or more?': random.choice(['Yes', 'No', 'No', 'Prefer not to say']),
        'Do any of your conditions or illnesses reduce your ability to carry out day-to-day activities?': random.choice(['Yes, a lot', 'Yes, a little', 'Not at all', 'Prefer not to say']),
        'What is your ethnic group? Please choose one option that best describes your ethnic group or background': random.choice(['White', 'Asian', 'Black', 'Mixed', 'Other']),
        'Please select your age': random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
        'Please select your gender': random.choice(['Male', 'Female', 'Prefer to self-describe', 'Prefer not to say']),
        'Over the last 4 weeks, how often have you done at least 30 minutes of moderate exercise (enough to raise your breathing rate)?': random.choice(['Not at all', '1-2 times', '3-4 times', '5+ times']),
        'Over the last 12 months, roughly how many times have you participated at parkrun as a volunteer?': random.choice(['Once', 'Between two and five times', 'Six or more times']),
        'AthleteID': random.randint(100000, 999999),
    }
    return row

# Generate volunteer survey data
print("Generating UK Volunteer Experience Survey data (500 rows)...")
volunteer_survey_path = r'C:\Users\olive\OneDrive\Documents\GitHub\parkrun_survey_analysis\data\surveys\UK Volunteer Experience Survey Blank Data - 15.5.26.csv'

# Get headers from the first generated row
first_row = generate_volunteer_survey_row(1)
headers = list(first_row.keys())

with open(volunteer_survey_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for i in range(1, 501):
        row = generate_volunteer_survey_row(i)
        writer.writerow(row)

print("Generated 500 rows for UK Volunteer Experience Survey")
print("\nMock data generation complete!")
