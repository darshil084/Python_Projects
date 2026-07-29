import pandas as pd
from scipy import stats
import plotly.express as px

data = pd.read_csv("Employee_Training_and_Performance_Dataset.csv")
#print(data.head())

group_yes = data[data['TrainingAttended']=='Yes']['PerformanceScore']
group_no = data[data['TrainingAttended']=='No']['PerformanceScore']

sample_size = min(len(group_yes), len(group_no), 300)
shapiro_yes = stats.shapiro(group_yes.sample(sample_size, random_state=1))
shapiro_no = stats.shapiro(group_no.sample(sample_size, random_state=1))

print("shapiro yes:", shapiro_yes)
print("shapiro no:", shapiro_no)

levene_test = stats.levene(group_yes, group_no)
print("levene test:", levene_test)

t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
print("t-statistic:", t_stat)
print("p-value:", p_value)

fig = px.box(
    data,
    x='TrainingAttended',
    y='PerformanceScore',
    title='Performance Score by Training Attendance',
    labels={
        'TrainingAttended': 'Training Attended',
        'PerformanceScore': 'Performance Score'
    },
    color='TrainingAttended',  
    points='all',  
)


fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    margin = dict(l=40, r=40, t=80, b=60),
    showlegend=False
)

fig.show()