python temporal_lcn_ir.py -s /media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/test_list_hand_sample.txt -d /media/DATA/LINUX_DATA/real_data_v10_temp -p 11 -t 0.2
python temporal_ir.py -s /media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/test_list_hand_sample.txt -d /media/DATA/LINUX_DATA/real_data_v10_temp -p 11 -t 0.005
python binary_ir.py -s /media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/test_list_hand_sample.txt -d /media/DATA/LINUX_DATA/real_data_v10_temp -p 11 -t 0.005
python lcn_ir.py -s /media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/test_list_hand_sample.txt -d /media/DATA/LINUX_DATA/real_data_v10_temp -p 11
