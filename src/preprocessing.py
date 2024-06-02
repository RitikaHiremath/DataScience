import os
import glob 
import pandas as pd

pathOK= "/Users/ritikahiremath/Downloads/Data/OK_Measurements/"

df_ok1=pd.DataFrame();
df_ok2=pd.DataFrame();

os.chdir(pathOK)

ok_list = glob.glob(pathOK + "/*")

all_okres1=[]
all_okres2=[]

for i in range(0,len(ok_list)): 
    df_oktemp1=pd.read_parquet(ok_list[i] + "/raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet")
    df_oktemp2=pd.read_parquet(ok_list[i] + "/raw/Sampling2000KHz_AEKi-0.parquet")
    all_okres1.append(df_oktemp1)
    all_okres2.append(df_oktemp2)
df_ok1=pd.concat(all_okres1)
df_ok2=pd.concat(all_okres2)

pathNOK="/Users/ritikahiremath/Downloads/Data/NOK_Measurements/"

os.chdir(pathNOK)

nok_list = glob.glob(pathNOK + "/*")

df_nok1=pd.DataFrame();
df_nok2=pd.DataFrame();

all_nokres1=[]
all_nokres2=[]

for i in range(0,len(nok_list)): 
    dfnok_temp1=pd.read_parquet(nok_list[i] + "/raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet")
    dfnok_temp2=pd.read_parquet(nok_list[i] + "/raw/Sampling2000KHz_AEKi-0.parquet")
    all_nokres1.append(dfnok_temp1)
    all_nokres2.append(dfnok_temp2)
df_nok1=pd.concat(all_nokres1)
df_nok2=pd.concat(all_nokres2)
