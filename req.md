I would like to propose the following topic for out three person group project(Yehor Karabanov, Denys Shevchenko, Ihor Tymkiv):
Lightweight ML-Based DDoS Detection System

The main goal of the project is to create a lightweight system that detects DDoS attacks using machine learning models and applies simple mitigation methods.

The idea is to analyze network traffic statistics and classify traffic as either normal or malicious.

I would also like to explore the following research questions:

Can lightweight ML models detect DDoS attacks accurately?

Which traffic features are the most useful?

How does ML-based detection compare to rule-based methods?
 
what types of DDoS attacks do you plan to identify? 
 
SYN flood
UDP flood
 
Fine. just make sure its DDoS you are working with, as single source DoS of this type are trival to identify


Dataset:

C:\ProgramData\chocolatey\bin\uv.exe run D:/Programming/DDoSED/.venv/Scripts/python.exe D:\Programming\DDoSED\src\load_dataset.py 
             Timestamp  ...                            Labels
0  2019-01-01 00:00:00  ...  Normal, Application Layer Attack
1  2019-01-01 01:00:00  ...                            Normal
2  2019-01-01 02:00:00  ...                            Normal
3  2019-01-01 03:00:00  ...                            Normal
4  2019-01-01 04:00:00  ...                            Normal

[5 rows x 24 columns]
(48192, 24)
Timestamp                       str
Source IP                       str
Destination IP                  str
Source Port                   int64
Destination Port              int64
Protocol                        str
Packet Size                   int64
Payload Length                int64
Flow Duration               float64
Bytes in Flow                 int64
Packets in Flow               int64
Average Packet Size         float64
Inter-Arrival Time          float64
Rate of Packets             float64
Unique Source Count           int64
Unique Destination Count      int64
Anomaly Score               float64
Device Type                     str
Operating System                str
Firmware Version                str
Attack Type                     str
Attack Duration             float64
Target Device                   str
Labels                          str
dtype: object

Process finished with exit code 0
