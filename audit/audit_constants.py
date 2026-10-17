"""Task constants for the standalone audit tool (copied from the project repo)."""

RISK_CLASSES = ('Indicator', 'Ideation', 'Behavior', 'Attempt')
FACTORS_24 = ('hopelessness', 'emotion dysregulation', 'poor social support', 'low self-esteem', 'suicide means (with access)', 'mental health issues', 'prior self-harm or suicidal thought/attempt', 'interpersonal difficulty', 'stressful life event', 'dysfunctional family', 'interpersonal violence', 'physical health/characteristic', 'traumatic experience', 'low socio-economic status', 'cognitive deficits', 'substance use', 'poor school performance', "exposure to others' suicide", 'sexual orientation related issues', 'coping strategy', 'psychological capital', 'social support', 'sense of responsibility', 'meaning in life')
RARE_FACTORS = ('cognitive deficits', 'substance use', 'poor school performance', "exposure to others' suicide", 'sexual orientation related issues', 'meaning in life')
FACTOR_DEFINITIONS = {
    'hopelessness': 'Feelings of hopelessness, or feeling trapped/stuck',
    'emotion dysregulation': 'Emotion regulation difficulties (e.g., uncontrolled anxiety, anger)',
    'poor social support': 'Lack friends, loneliness, being isolated/rejected/neglected/abandoned',
    'low self-esteem': 'Self-negative feelings (e.g., worthlessness, being a burden, self-hate)',
    'suicide means (with access)': 'Description of potential means or access to that means',
    'mental health issues': 'Existing mental disorders (e.g., depression, personality disorders)',
    'prior self-harm or suicidal thought/attempt': 'Past, resolved, previous self-harm and suicidal thought/plan/attempt',
    'interpersonal difficulty': 'Having difficulty making new friends, socialization, etc.',
    'stressful life event': 'Events that pose challenges but do not lead to traumatic responses',
    'dysfunctional family': 'Family-related issues that lead to negative impacts on the individual',
    'interpersonal violence': 'Violence or assault, occurring outside home settings',
    'physical health/characteristic': 'Physical health issues (e.g., COVID-19, obesity/underweight)',
    'traumatic experience': "Event(s) that is overwhelming an individual's ability to cope",
    'low socio-economic status': 'Unemployment status, poverty, and homeless, etc.',
    'cognitive deficits': 'Having difficulty in cognitive abilities',
    'substance use': 'The uncontrollable use of drugs/alcohol/tobacco',
    'poor school performance': 'Low school performance (e.g., failing tests, bad grades)',
    "exposure to others' suicide": "Mentioning or describing others' suicidal thoughts, attempt or death",
    'sexual orientation related issues': 'Having gender/sexual disorder, same-sex relationship, etc.',
    'coping strategy': 'Activities individuals engage in to deal with stressful situations',
    'psychological capital': "Individual's positive psychological state of development",
    'social support': 'Support from family, partners, friends, healthcare professionals',
    'sense of responsibility': "Awareness of responsibility to one's own health/survival and to others",
    'meaning in life': 'Involves cognitive, motivational and affective components',
}
