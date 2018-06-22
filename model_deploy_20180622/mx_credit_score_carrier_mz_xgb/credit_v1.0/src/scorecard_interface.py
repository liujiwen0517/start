# -*- coding : utf-8 -*-
import requests
import json


#kg 
# url = "http://121.43.63.113:6033/api/carrier/risk/score/v1.0"

# local
url = "http://0.0.0.0:6037/api/carrier_mz/risk/score/v1.0"

payload = '{"carrier_recharge_freq_amount_over50yuan_150d":1.0,"carrier_call_ratio_freq_record_telephone_dialed_150d":0.0,"carrier_call_ratio_freq_record_work_dialed_150d":0.22,"carrier_call_ratio_freq_record_nowork_dial_150d":0.11,"carrier_call_ratio_cnt_peernumber_night_dail_150d":0.03,"carrier_call_cnt_peer_province_dialed_150d":9.0,"carrier_call_freq_diff_province_all_150d":534.0,"carrier_call_cnt_peernumber_telephone_dialed_150d":2.0,"carrier_call_ratio_cnt_peernumber_telephone_dialed_150d":0.01,"carrier_call_freq_peopletype_agency_dailed_150d":1.0,"carrier_call_freq_peopletype_deliver_dailed_150d":8.0,"carrier_call_freq_peopletype_tagged_dailed_150d":0.0,"carrier_call_freq_peopletype_crank_dailed_150d":0.0,"carrier_call_freq_durationinsecond_most_10s_150d":114.0,"carrier_call_ratio_freq_durationinsecond_most_10s_150d":0.08,"carrier_call_ratio_freq_days_dialed_150d":0.82,"carrier_call_freq_days_work_dialed_150d":112.0,"carrier_call_ratio_freq_durationinsecond_60s_300s_150d":0.33,"carrier_call_avg_record_sum_durationinsecond_dialed_150d":106.08,"carrier_call_cnt_peernumber_day_dailed_150d":104.0,"carrier_call_ratio_cnt_peernumber_day_dailed_150d":0.61,"carrier_call_ratio_freq_record_shangwu_dialed_150d":0.06,"carrier_call_ratio_cnt_peernumber_shangwu_dailed_150d":0.28,"carrier_call_freq_record_zhongwu_dialed_150d":41.0,"carrier_call_ratio_cnt_peernumber_zhongwu_dail_150d":0.18,"carrier_call_cnt_peernumber_bangwan_dailed_150d":29.0,"carrier_call_ratio_cnt_peernumber_bangwan_dailed_150d":0.17,"carrier_call_ratio_cnt_peernumber_bangwan_dail_150d":0.16,"carrier_call_ratio_cnt_peernumber_wanshang_dailed_150d":0.11,"carrier_call_ratio_cnt_peernumber_wanshang_dail_150d":0.11,"carrier_call_ratio_cnt_peernumber_lingchen_dailed_150d":0.02,"carrier_call_cnt_peernumber_diff_city_all_150d":34.0,"carrier_call_cnt_roaming_peernumber_150d":10.0,"carrier_call_freq_peopletype_all_150d":68.0,"carrier_call_freq_peopletype_dialed_150d":10.0,"carrier_call_ratio_peernumber_dial_3m_150d":0.17,"carrier_call_ratio_peernumber_dialed_3m_150d":0.13,"province":"\\u5185\\u8499\\u53e4","blackpeercnt":0.0,"indirectpeercnt":15.0,"close_blackpeercnt":0.0,"close_indirectpeercnt":5.0,"debit_balance":null,"maxtotalcreditlimit":10000.0,"delaybillmonth":0.0,"cash_loan_180d":37.0,"diversion_90d":0.0,"diversion_180d":0.0,"datacoverge_180d":2.0,"consumstage_90d":0.0,"consumstage_180d":0.0,"name":"\\u738b\\u5316\\u6960","mobile":"18158114141","idcard":"232128198711153410"}'


# 缺少三要素中的一个或多个
# 少id 如idcard
# payload = '{"carrier_call_freq_days_work_all_150d":123.0,"carrier_call_ratio_cnt_peernumber_wanshang_dail_150d":0.12,"carrier_sms_cnt_peercity_out_150d":7.0,"carrier_call_freq_peopletype_tourism_all_150d":1.0,"carrier_bill_sum_basefee_150d":450.0,"carrier_recharge_avg_amount_single_150d":32.74,"carrier_sms_cnt_peercity_150d":13.0,"carrier_call_cnt_peernumber_dialed_3m_150d":32.0,"carrier_call_freq_days_work_dialed_150d":116.0,"carrier_sms_cnt_peernumber_midnight_out_150d":2.0,"carrier_sms_ratio_freq_record_in_work_150d":0.68,"carrier_call_freq_peopletype_crank_all_150d":29.0,"carrier_call_cnt_peernumber_same_province_all_150d":188.0,"carrier_sms_cnt_peernumber_midnight_in_150d":59.0,"carrier_call_sum_durationinsecond_night_dialed_150d":128.0,"carrier_call_freq_days_weekday_dialed_150d":81.0,"carrier_call_cnt_peernumber_60s_150d":134.0,"carrier_call_freq_peopletype_deliver_all_150d":1.0,"carrier_call_ratio_freq_days_work_dialed_150d":0.91,"carrier_call_cnt_peer_province_all_150d":18.0,"carrier_call_freq_days_all_150d":128.0,"carrier_call_ratio_peernumber_dialed_3m_150d":0.11,"carrier_call_ratio_freq_days_night_dial_150d":0.02,"carrier_call_freq_days_dialed_150d":120.0,"carrier_call_max_cnt_days_samenumber_work_all_150d":19.0,"carrier_sms_freq_other_in_150d":258.0,"carrier_call_max_cnt_day_samenumber_all_150d":25.0,"carrier_call_ratio_cnt_peernumber_night_dail_150d":0.01,"carrier_call_freq_peopletype_carrier_dail_150d":9.0,"carrier_call_freq_peopletype_carrier_all_150d":10.0,"carrier_call_ratio_cnt_peernumber_shenye_dail_150d":0.01,"carrier_sms_cnt_peernumber_out_150d":32.0,"carrier_call_ratio_cnt_peernumber_mobile_dial_150d":0.43,"carrier_call_cnt_peernumber_diff_province_all_150d":80.0,"carrier_base_level":"\\u4e09\\u661f","carrier_call_freq_peopletype_tagged_dailed_150d":6.0,"carrier_call_freq_peopletype_crank_dailed_150d":29.0,"carrier_call_sum_durationinsecond_telephone_dialed_150d":2563.0,"carrier_call_cnt_peernumber_same_city_all_150d":188.0,"carrier_call_freq_days_holiday_all_150d":42.0,"carrier_call_freq_peopletype_tourism_dailed_150d":1.0,"carrier_base_availablebalance":59.99,"carrier_bill_min_actualfee_150d":106.2,"carrier_sms_ratio_freq_record_in_night_150d":0.05,"carrier_call_max_cnt_day_samenumber_dail_150d":19.0,"carrier_call_ratio_cnt_peernumber_mobile_dialed_150d":0.46,"carrier_sms_avg_cnt_peernumber_midnight_monthly_out_150d":0.4,"carrier_call_freq_days_weekday_all_150d":86.0,"carrier_call_cnt_days_silence_all_150d":22.0,"carrier_call_ratio_freq_record_nowork_dialed_150d":0.08,"carrier_call_cnt_peer_province_dialed_150d":16.0,"carrier_bill_max_basefee_150d":98.0,"carrier_call_freq_peopletype_deliver_dailed_150d":1.0,"carrier_bill_min_totalfee_150d":116.2,"carrier_call_freq_peernumber_5s_150d":6.0,"carrier_call_ratio_freq_durationinsecond_60s_300s_150d":0.36,"carrier_base_length_time_innet":1197.0,"name":"\\u738b\\u5316\\u6960","mobile":"18158114141"}'


# 特征缺失
# 少一个特征
# payload = '{"carrier_call_ratio_cnt_peernumber_wanshang_dail_150d":0.12,"carrier_sms_cnt_peercity_out_150d":7.0,"carrier_call_freq_peopletype_tourism_all_150d":1.0,"carrier_bill_sum_basefee_150d":450.0,"carrier_recharge_avg_amount_single_150d":32.74,"carrier_sms_cnt_peercity_150d":13.0,"carrier_call_cnt_peernumber_dialed_3m_150d":32.0,"carrier_call_freq_days_work_dialed_150d":116.0,"carrier_sms_cnt_peernumber_midnight_out_150d":2.0,"carrier_sms_ratio_freq_record_in_work_150d":0.68,"carrier_call_freq_peopletype_crank_all_150d":29.0,"carrier_call_cnt_peernumber_same_province_all_150d":188.0,"carrier_sms_cnt_peernumber_midnight_in_150d":59.0,"carrier_call_sum_durationinsecond_night_dialed_150d":128.0,"carrier_call_freq_days_weekday_dialed_150d":81.0,"carrier_call_cnt_peernumber_60s_150d":134.0,"carrier_call_freq_peopletype_deliver_all_150d":1.0,"carrier_call_ratio_freq_days_work_dialed_150d":0.91,"carrier_call_cnt_peer_province_all_150d":18.0,"carrier_call_freq_days_all_150d":128.0,"carrier_call_ratio_peernumber_dialed_3m_150d":0.11,"carrier_call_ratio_freq_days_night_dial_150d":0.02,"carrier_call_freq_days_dialed_150d":120.0,"carrier_call_max_cnt_days_samenumber_work_all_150d":19.0,"carrier_sms_freq_other_in_150d":258.0,"carrier_call_max_cnt_day_samenumber_all_150d":25.0,"carrier_call_ratio_cnt_peernumber_night_dail_150d":0.01,"carrier_call_freq_peopletype_carrier_dail_150d":9.0,"carrier_call_freq_peopletype_carrier_all_150d":10.0,"carrier_call_ratio_cnt_peernumber_shenye_dail_150d":0.01,"carrier_sms_cnt_peernumber_out_150d":32.0,"carrier_call_ratio_cnt_peernumber_mobile_dial_150d":0.43,"carrier_call_cnt_peernumber_diff_province_all_150d":80.0,"carrier_base_level":"\\u4e09\\u661f","carrier_call_freq_peopletype_tagged_dailed_150d":6.0,"carrier_call_freq_peopletype_crank_dailed_150d":29.0,"carrier_call_sum_durationinsecond_telephone_dialed_150d":2563.0,"carrier_call_cnt_peernumber_same_city_all_150d":188.0,"carrier_call_freq_days_holiday_all_150d":42.0,"carrier_call_freq_peopletype_tourism_dailed_150d":1.0,"carrier_base_availablebalance":59.99,"carrier_bill_min_actualfee_150d":106.2,"carrier_sms_ratio_freq_record_in_night_150d":0.05,"carrier_call_max_cnt_day_samenumber_dail_150d":19.0,"carrier_call_ratio_cnt_peernumber_mobile_dialed_150d":0.46,"carrier_sms_avg_cnt_peernumber_midnight_monthly_out_150d":0.4,"carrier_call_freq_days_weekday_all_150d":86.0,"carrier_call_cnt_days_silence_all_150d":22.0,"carrier_call_ratio_freq_record_nowork_dialed_150d":0.08,"carrier_call_cnt_peer_province_dialed_150d":16.0,"carrier_bill_max_basefee_150d":98.0,"carrier_call_freq_peopletype_deliver_dailed_150d":1.0,"carrier_bill_min_totalfee_150d":116.2,"carrier_call_freq_peernumber_5s_150d":6.0,"carrier_call_ratio_freq_durationinsecond_60s_300s_150d":0.36,"carrier_base_length_time_innet":1197.0,"idcard":"232128198711153410","name":"\\u738b\\u5316\\u6960","mobile":"18158114141"}'





headers = {
    'content-type': "application/json"
    }


response = requests.request("POST",url,data=payload,headers=headers)
print(response.text)
