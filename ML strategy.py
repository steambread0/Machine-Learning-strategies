# get US T-bond future data
import quandl
from datetime import datetime
import numpy,pandas
import math
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Download Treasury bond future(front month) data
data=quandl.get("CHRIS/CME_US1")

# Download 30 year T-bond yield data
t_yield=quandl.get("USTREASURY/YIELD")
yield_30=t_yield['30 YR']


# Processing data
# We choose data from 2010-01-01 to 2018-12-31
row=0
while True:
    if datetime.fromtimestamp(datetime.timestamp(data.index[row])).year==2010:
        start=row
        row+=300
    elif datetime.fromtimestamp(datetime.timestamp(data.index[row])).year==2019:
        end=row-1
        break
    else:
        row+=1
sample_data=data.ix[start:end]

row=0
while True:
    if datetime.fromtimestamp(datetime.timestamp(yield_30.index[row])).year==2010:
        start=row
        row+=300
    elif datetime.fromtimestamp(datetime.timestamp(yield_30.index[row])).year==2019:
        end=row-1
        break
    else:
        row+=1
sample_yield=yield_30[start:end]


# Get the date in string format
date=list(data.index)
date=[str(date[i])[:10] for i in range(len(date))]

# Define a function to find the index of a specific date
def find_position(s,l):
    result=[]
    for i in range(len(l)):
        result.append(s in l[i])
    pos=result.index(True)
    return pos

# Define a function to compute drawdown
def calculate_drawdown(dt):
    drawdown=max(max(dt)/dt[-1]-1,0)
    return drawdown

# Define a function to compute sortino ratio
def calculate_daily_sortino_ratio(dt):
    daily_return=numpy.array(dt[1:])/numpy.array(dt[:-1])-1
    downside_deviation=0
    for i in range(len(daily_return)):
        downside_deviation+=min(daily_return[i],0)**2
    downside_deviation=numpy.sqrt(downside_deviation/len(daily_return))
    expected_return=numpy.mean(daily_return)
    sortino_ratio=expected_return/downside_deviation
    return sortino_ratio

# Define a function to compute return
def calculate_return(dt):
    return dt[-1]-dt[0]


# Compute real monthly return, sortino ratio and drawdown
# They are used in as future Machine Learning input

dd_real=[]
ret_real=[]
sr_real=[]
year=2010
month=1
initial_index=find_position('%s-01-'%year,date)
while year!=2018 or month!=12:
    if month==12:
        begin=find_position('%s-12-'%year,date)-initial_index
        end=find_position('%s-01-'%(year+1),date)-1-initial_index
    elif month==10 or month==11:
        begin=find_position('%s-%s-'%(year,month),date)-initial_index
        end=find_position('%s-%s-'%(year,month+1),date)-1-initial_index
    elif month==9:
        begin=find_position('%s-0%s-'%(year,month),date)-initial_index
        end=find_position('%s-%s-'%(year,month+1),date)-1-initial_index
    else:
        begin=find_position('%s-0%s-'%(year,month),date)-initial_index
        end=find_position('%s-0%s-'%(year,month+1),date)-1-initial_index
    dd_real.append(calculate_drawdown(sample_data['Last'][begin:end]))
    ret_real.append(calculate_return(sample_data['Last'][begin:end]))
    sr_real.append(calculate_daily_sortino_ratio(sample_data['Last'][begin:end]))
    month+=1
    if month==13:
        year+=1
        month-=12

# Fill nan in those three 
for i in range(len(dd_real)):
    if math.isnan(dd_real[i]):
        dd_real[i]=0
    elif math.isnan(ret_real[i]):
        ret_real[i]=0
    elif math.isnan(sr_real[i]):
        sr_real[i]=0

# Use Machine Learning to predict the future price

'''We use price, volume and yield of last month as input to predict price, sortino ratio and drawdown of 
the first day of the next month'''

year=2010
month=1
row=0
count=0
start=row
whole_x=[]
whole_y=[]
startend=[]
while year!=2018 or month!=12:
    if datetime.fromtimestamp(datetime.timestamp(sample_data.index[row])).month==12:
        if datetime.fromtimestamp(datetime.timestamp(sample_data.index[row+1])).month==1:
            end=row+1
            whole_y.append([sample_data['Last'][end]]+[ret_real[count],dd_real[count],sr_real[count]])
            count+=1
            volume=list(numpy.array(sample_data['Volume'][start:end])/100000)
            startend.append([start,end])
            if len(sample_data['Last'][start:end])<23:
                price=list(sample_data['Last'][start:end])+[numpy.mean(sample_data['Last'][start:end])]*(23-len(sample_data['Last'][start:end]))
                yld=list(sample_yield[start:end])+[numpy.mean(sample_yield[start:end])]*(23-len(sample_yield[start:end]))
                vol=volume+[numpy.mean(volume)]*(23-len(volume))
            else:
                price=list(sample_data['Last'][start:end])
                yld=list(sample_yield[start:end])
                vol=volume
            whole_x.append(price+vol+yld)
            start=end
            year+=1
            month=1
    elif datetime.fromtimestamp(datetime.timestamp(sample_data.index[row+1])).month==month+1:
        end=row+1
        whole_y.append([sample_data['Last'][end]]+[ret_real[count],dd_real[count],sr_real[count]])
        count+=1
        volume=list(numpy.array(sample_data['Volume'][start:end])/100000)
        startend.append([start,end])
        if len(sample_data['Last'][start:end])<23:
                price=list(sample_data['Last'][start:end])+[numpy.mean(sample_data['Last'][start:end])]*(23-len(sample_data['Last'][start:end]))
                yld=list(sample_yield[start:end])+[numpy.mean(sample_yield[start:end])]*(23-len(sample_yield[start:end]))
                vol=volume+[numpy.mean(volume)]*(23-len(volume))
        else:
            price=list(sample_data['Last'][start:end])
            yld=list(sample_yield[start:end])
            vol=volume
        whole_x.append(price+vol+yld)
        start=end
        month+=1
    row+=1

# Processing nan in whole_x
for i in range(len(whole_x)):
    for j in range(len(whole_x[i])):
        if math.isnan(whole_x[i][j]):
            whole_x[i][j]=whole_x[i][j-1]

# Processing nan in whole_y
for i in range(len(whole_y)):
    for j in range(len(whole_y[i])):
        if math.isnan(whole_y[i][j]) and j==0:
            whole_y[i][j]=whole_y[i-1][j]
        elif math.isnan(whole_y[i][j]) and j>=1:
            whole_y[i][j]=0


# Train the training set and test
# 70% train and 30% test

# Devide training set and test set
trainx=whole_x[:int(numpy.ceil(len(whole_x)*0.7))]
testx=whole_x[int(numpy.ceil(len(whole_x)*0.7)):]
trainy=whole_y[:int(numpy.ceil(len(whole_x)*0.7))]
testy=whole_y[int(numpy.ceil(len(whole_x)*0.7)):]

# Normalize data
sc = StandardScaler()
sc.fit(trainx)
trainxstd = sc.transform(trainx)
testxstd = sc.transform(testx)

# Predict future price using neural network
rgs=MLPRegressor(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(4, 5),max_iter=300,tol=1e-3)
rgs.fit(trainxstd,trainy)
predy=rgs.predict(testxstd)

# Result format: [price,ret,dd,sr]
# You can print the predicted price and real price
#py=pandas.DataFrame(predy)[0]
#ry=pandas.DataFrame(testy)[0]
#pandas.DataFrame({'pred':py,'real':ry})


## Build Strategy
## We want to max(return+sortino_ratio-drawdown)

# Predicted 
long_short_increment_pred=[]
for i in range(len(predy)):
    if i==1:
        month_ret=predy[i][0]-trainy[-1][0]
    else:
        month_ret=predy[i][0]-predy[i-1][0]                                   #### revised
    long_incre=round(month_ret+predy[i][3]-predy[i][2],3)
    short_incre=round(-month_ret+predy[i][3]-predy[i][2],3)
    long_short_increment_pred.append([long_incre,short_incre])

# Real
long_short_increment_test=[]
for i in range(len(testy)):
    if i==1:
        month_ret=testy[i][0]-testy[-1][0]
    else:
        month_ret=testy[i][0]-testy[i-1][0]
    long_incre=round(month_ret+testy[i][3]-testy[i][2],3)
    short_incre=round(-month_ret+testy[i][3]-testy[i][2],3)
    long_short_increment_test.append([long_incre,short_incre])

# Compare predict value and real value of return+sortino-drawdown
# Format: [change if long,change if short]
#pandas.DataFrame({'pred':long_short_increment_pred,'real':long_short_increment_test})


## Next we use Dynamic Programming to find the optimized long-short strategy
'''value_mat: Used to store long-short increment at each node
   result_mat: Used to store longest length from the last node to current node
   path_mat: Used to store the longest path'''

# Initialize these three matrixes
selected=len(predy)
value_list=long_short_increment_pred[:selected]
value_mat=pandas.DataFrame([[[0,0]]*selected]*(selected+3))
result_mat=pandas.DataFrame(numpy.ones([selected+3,selected+1])*(-10000))
result_mat.iloc[int(selected/2+1),selected]=0
path_mat=pandas.DataFrame(numpy.zeros([selected+3,selected+1]))

# Fill the value_mat with predicted values
for_timee=0
first_row=int(0.5*selected+2)
for i in range(selected):
    if i<=0.5*selected:
        first_row-=1
        for_timee+=1
    else:
        first_row+=1
        for_timee-=1
    current_row=first_row
    j=0
    while j<for_timee:
        value_mat.iloc[current_row,i]=value_list[i]
        current_row+=2
        j+=1

# Print value_mat
#value_mat


# Test the algorithm, DP beginning from the last node
first_col=selected
first_row=int(0.5*selected+1)
count=0
current_row=first_row
current_col=first_col
for_times=1
while current_col!=0 and count!=selected:
    current_col-=1
    count+=1
    if count<=0.5*selected:
        first_row=first_row-1
        for_times+=1
    else:
        first_row=first_row+1
        for_times-=1
    current_row=first_row
    for i in range(for_times):
        result_mat.iloc[current_row,current_col]=max(result_mat.iloc[current_row+1,current_col+1]+value_mat.iloc[current_row,current_col][1],result_mat.iloc[current_row-1,current_col+1]+value_mat.iloc[current_row,current_col][0])
        if result_mat.iloc[current_row,current_col]==result_mat.iloc[current_row+1,current_col+1]+value_mat.iloc[current_row,current_col][1]:
            path_mat.iloc[current_row,current_col]=1
        else:
            path_mat.iloc[current_row,current_col]=-1
        #print(current_row,current_col)
        current_row+=2
    current_row=first_row

# Print the result_mat and path_mat
# result_mat
# path_mat

# Print the path, which is our strategy
c_row=int(0.5*selected+1)
c_col=0
init_value=200000
i=0
path=[]
cap_value=[init_value]
while c_col!=selected:
    if path_mat.iloc[c_row,c_col]==1:
        c_row+=1
        path.append('short')
        init_value+=100*predy[i][0]
    elif path_mat.iloc[c_row,c_col]==-1:
        c_row-=1
        path.append('long')
        init_value-=100*predy[i][0]
    cap_value.append(init_value)
    i+=1
    c_col+=1
print(path)

# Plot the cap_value
import matplotlib.pyplot as plt
plt.plot(cap_value,color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Capital value')
plt.axhline(y=200000, color='green', linestyle='--')
plt.show()
print('Current capital:',init_value)