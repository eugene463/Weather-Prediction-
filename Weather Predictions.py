#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

weather = pd.read_csv("local_weather.csv", index_col="DATE")


# In[4]:


weather


# In[5]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[6]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[7]:


core_weather.apply(pd.isnull).sum()


# In[8]:


core_weather["snow"].value_counts()


# In[9]:


core_weather["snow_depth"].value_counts()


# In[10]:


del core_weather["snow"]


# In[11]:


del core_weather["snow_depth"]


# In[12]:


core_weather[pd.isnull(core_weather["precip"])]


# In[13]:


core_weather.loc["2013-12-15",:]


# In[14]:


core_weather["precip"].value_counts() / core_weather.shape[0]


# In[15]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[16]:


core_weather.apply(pd.isnull).sum()


# In[17]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[19]:


core_weather.loc["2011-12-18":"2011-12-28"]


# In[20]:


core_weather = core_weather.fillna(method="ffill")


# In[21]:


core_weather.apply(pd.isnull).sum()


# In[22]:


# Check for missing value defined in data documentation
core_weather.apply(lambda x: (x == 9999).sum())


# In[23]:


core_weather.dtypes


# In[24]:


core_weather.index


# In[25]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[26]:


core_weather.index


# In[27]:


core_weather.index.year


# In[28]:


core_weather[["temp_max", "temp_min"]].plot()


# In[29]:


core_weather.index.year.value_counts().sort_index()


# In[30]:


core_weather["precip"].plot()


# In[31]:


core_weather.groupby(core_weather.index.year).apply(lambda x: x["precip"].sum()).plot()


# In[32]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[33]:


core_weather


# In[34]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[35]:


core_weather


# In[36]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)


# In[37]:


predictors = ["precip", "temp_max", "temp_min"]


# In[38]:


train = core_weather.loc[:"2020-12-31"]
test = core_weather.loc["2021-01-01":]


# In[39]:


train


# In[40]:


test


# In[41]:


reg.fit(train[predictors], train["target"])


# In[42]:


predictions = reg.predict(test[predictors])


# In[43]:


from sklearn.metrics import mean_squared_error

mean_squared_error(test["target"], predictions)


# In[44]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[45]:


combined


# In[46]:


combined.plot()


# In[47]:


reg.coef_


# In[48]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()

core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]

core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[53]:


core_weather = core_weather.iloc[30:,:].copy()


# In[54]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_squared_error(test["target"], predictions)
    
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[55]:


predictors = ["precip", "temp_max", "temp_min", "month_day_max", "max_min"]

error, combined = create_predictions(predictors, core_weather, reg)
error


# In[56]:


combined.plot()


# In[58]:


core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month).apply(lambda x: x.expanding(1).mean())
core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year).apply(lambda x: x.expanding(1).mean())


# In[59]:


error, combined = create_predictions(predictors + ["monthly_avg", "day_of_year_avg"], core_weather, reg)
error


# In[60]:


reg.coef_


# In[61]:


core_weather.corr()["target"]


# In[62]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[63]:


combined.sort_values("diff", ascending=False).head(10)


# In[ ]:




