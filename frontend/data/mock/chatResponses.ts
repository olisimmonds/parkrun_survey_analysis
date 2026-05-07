import type { Source } from '@/types/chat';

interface MockResponse {
  content: string;
  sources: Source[];
}

export const mockResponses: MockResponse[] = [
  {
    content: `Based on the **Volunteer Experience Survey 2024**, the top three factors driving volunteer satisfaction are:

1. **Sense of community** — 78% of volunteers cited belonging to a welcoming community as their primary motivator
2. **Personal health and fitness** — 64% reported that volunteering keeps them active even when not running
3. **Making a difference** — 61% feel their contribution has a tangible positive impact on participants

Volunteers aged 55+ showed significantly higher satisfaction scores (avg. 4.6/5) compared to younger volunteers (avg. 4.1/5), suggesting the programme resonates most strongly with older community members.

**Areas for improvement** highlighted included clearer communication of role expectations (mentioned by 32% of respondents) and more recognition for long-serving volunteers (27%).`,
    sources: [
      {
        id: 'src_001',
        name: 'Volunteer Experience Survey 2024',
        excerpt: 'Q14: "What is your primary motivation for volunteering at parkrun?" — Community (78%), Health (64%), Making a difference (61%)',
        datasetId: 'ds_001',
        relevanceScore: 0.97,
      },
      {
        id: 'src_002',
        name: 'Volunteer Experience Survey 2024',
        excerpt: 'Satisfaction by age group: 55+ avg 4.6/5, 35-54 avg 4.3/5, under-35 avg 4.1/5',
        datasetId: 'ds_001',
        relevanceScore: 0.91,
      },
    ],
  },
  {
    content: `The **Participant Wellbeing Survey Q3 2024** reveals strong positive mental health outcomes from regular parkrun participation:

- **89%** of respondents reported improved mood after attending parkrun events
- **73%** said parkrun has helped reduce feelings of isolation or loneliness
- Participants attending **4+ times per month** show significantly higher wellbeing scores compared to occasional participants
- **67%** of those who joined primarily for physical fitness now cite the social aspect as equally or more important

The data suggests a virtuous cycle: participants who feel socially connected attend more frequently, which correlates with greater reported wellbeing improvements.

> "parkrun has genuinely changed my life. I came for the running but stayed for the people." — Representative quote from open-text responses`,
    sources: [
      {
        id: 'src_003',
        name: 'Participant Wellbeing Survey Q3 2024',
        excerpt: 'Q8: "Has parkrun participation improved your mental wellbeing?" — Yes: 89%, No change: 9%, Worse: 2%',
        datasetId: 'ds_002',
        relevanceScore: 0.95,
      },
      {
        id: 'src_004',
        name: 'Participant Wellbeing Survey Q3 2024',
        excerpt: 'Frequency vs wellbeing correlation: r=0.67, p<0.001. High frequency (4+/month) mean score: 4.4/5',
        datasetId: 'ds_002',
        relevanceScore: 0.88,
      },
    ],
  },
  {
    content: `Analysing the **New Participant Onboarding Survey** alongside **Event Attendance Data 2024** reveals key patterns in participant retention:

**Discovery channels:**
- Word of mouth: 52%
- Social media: 28%
- GP / health professional referral: 11%
- Online search: 9%

**Retention risk factors** — participants who did not return after their first event most commonly cited:
- Felt intimidated by pace/fitness levels (41%)
- Didn't feel welcomed or noticed (29%)
- Logistical barriers (parking, time) (22%)

**Recommendations:**
1. Strengthen the first-timer experience with a dedicated welcome volunteer role
2. Ensure the finish funnel experience is warm and encouraging for slower finishers
3. Consider a follow-up communication within 48hrs of a first-time finish`,
    sources: [
      {
        id: 'src_005',
        name: 'New Participant Onboarding Survey',
        excerpt: 'Q3: "How did you first hear about parkrun?" responses — WOM: 52%, Social: 28%, GP: 11%, Search: 9%',
        datasetId: 'ds_004',
        relevanceScore: 0.93,
      },
      {
        id: 'src_006',
        name: 'Event Attendance Data 2024',
        excerpt: 'First-time participants who returned within 30 days: 61%. Did not return within 90 days: 22%',
        datasetId: 'ds_003',
        relevanceScore: 0.82,
      },
    ],
  },
];

export function getRandomResponse(): MockResponse {
  return mockResponses[Math.floor(Math.random() * mockResponses.length)];
}
