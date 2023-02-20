from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib
import pandas as pd


model_path = pickle.load(open('phishing.pkl', 'rb'))




def home():
    return render_template('home.html')


def predict():
    #print('test');
    text_URL = (request.form['text_URL'])
    print(text_URL)
    #Breathing_Problem = int(request.form['Breathing_Problem'])
    #print(pre)
    #df=pd.read_excel('./test.xlsx',header=None)
    #query = np.array([[Destination_Port,Flow_Duration,Total_Fwd_Packets,Total_Backward_Packets,Total_Length_of_Fwd_Packets,Total_Length_of_Bwd_Packets,Fwd_Packet_Length_Max,Fwd_Packet_Length_Min,Fwd_Packet_Length_Mean,Fwd_Packet_Length_Std,Bwd_Packet_Length_Max,Bwd_Packet_Length_Min,Bwd_Packet_Length_Mean,Bwd_Packet_Length_Std,Flow_Bytes_Sec,Flow_Packets_Sec,Flow_IAT_Mean,Flow_IAT_Std,Flow_IAT_Max,Flow_IAT_Min ,Fwd_IAT_Total,Fwd_IAT_Mean,Fwd_IAT_Std,Fwd_IAT_Max,Fwd_IAT_Min,Bwd_IAT_Total,Bwd_IAT_Mean,Bwd_IAT_Std ,Bwd_IAT_Max,Bwd_IAT_Min,Fwd_PSH_Flags,Bwd_PSH_Flags,Fwd_Header_Length,Bwd_Header_Length,Fwd_Packets_Sec,Bwd_Packets_Sec,Min_Packet_Length,Max_Packet_Length,Packet_Length_Mean,Packet_Length_Std,Packet_Length_Variance,FIN_Flag_Count,SYN_Flag_Count,RST_Flag_Count,PSH_Flag_Count,ACK_Flag_Count,URG_Flag_Count,Down_Up_Ratio,Average_Packet_Size,Avg_Fwd_Segment_Size,Avg_Bwd_Segment_Size,Subflow_Fwd_Packets,Subflow_Fwd_Bytes,Subflow_Bwd_Packets,Subflow_Bwd_Bytes,Init_Win_bytes_forward,Init_Win_bytes_backward,act_data_pkt_fwd,min_seg_size_forward,Active_Mean,Active_Std,Active_Max,Active_Min,Idle_Mean,Idle_Max,Idle_Min]])
   #query = np.array([[Destination_Port,Flow_Duration,Total_Fwd_Packets,Total_Backward_Packets	,Total_Length_of_Fwd_Packets,Total_Length_of_Bwd_Packets,Fwd_Packet_Length_Max,Fwd_Packet_Length_Min,Fwd_Packet_Length_Mean,Fwd_Packet_Length_Std,Bwd_Packet_Length_Max,Bwd_Packet_Length_Min,Bwd_Packet_Length_Mean,Bwd_Packet_Length_Std,Flow_Bytes_Sec,Flow_Packets_Sec,Flow_IAT_Mean,Flow_IAT_Std,Flow_IAT_Max,Flow_IAT_Min ,Fwd_IAT_Total,Fwd_IAT_Mean,Fwd_IAT_Std ,Fwd_IAT_Max,Fwd_IAT_Min,Bwd_IAT_Total,Bwd_IAT_Mean,Bwd_IAT_Std ,Bwd_IAT_Max,Bwd_IAT_Min,Fwd_PSH_Flags,Bwd_PSH_Flags,Fwd_Header_Length,Bwd_Header_Length,Fwd_Packets_Sec,Bwd_Packets_Sec,Min_Packet_Length,Max_Packet_Length,Packet_Length_Mean,Packet_Length_Std,Packet_Length_Variance,FIN_Flag_Count	,SYN_Flag_Count	,RST_Flag_Count	,PSH_Flag_Count	,ACK_Flag_Count	,URG_Flag_Count	,Down_Up_Ratio,Average_Packet_Size,Avg_Fwd_Segment_Size,Avg_Bwd_Segment_Size,Subflow_Fwd_Packets,Subflow_Fwd_Bytes,Subflow_Bwd_Packets,Subflow_Bwd_Bytes,Init_Win_bytes_forward,Init_Win_bytes_backward,act_data_pkt_fwd,min_seg_size_forward,Active_Mean,Active_Std,Active_Max ,Active_Min,Idle_Mean,Idle_Std,Idle_Max,Idle_Min]])
    #query = ([[text_URL]])


    prediction=model.predict([text_URL])
    #prediction = model.predict(['zimbio.com/Coco/articles/hgVpGi5atEj/Rapper+Ice+T+buxom+wife+Coco+reveals+breast'])#,'service.confirm.paypal.cmd.cgi-bin.2466sd4f3e6... '])
    #print(prediction)
   
    return render_template('result.html',prediction=prediction)
   

    
                       
    
    














