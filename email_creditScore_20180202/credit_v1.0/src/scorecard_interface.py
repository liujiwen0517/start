# -*- coding : utf-8 -*-
import requests
import json



# online 218
# url = "http://118.31.250.218:6031/api/email/email_creditScore/v1.0"

# online 129
# url = "http://120.27.195.129:6029/api/xigua/credit_score_carrier/v1.0"

# local server
url = "http://0.0.0.0:6033/api/email/email_creditScore/v1.0"



payload = '{"emailid":51236470812365968,"nameoncard":"DWO2re9ufo8=","mail_bill_min_ratio_nopay_6mth":-1.0,"mail_bill_avg_nopayment_6mth":-2090.28,"email_billitem_freq_overdue_6mth":0.0,"payamt_6":14420.0,"mail_card_sum_cashlimit_singlebank":20000.0,"mail_bill_avg_lastpayment_6mth":6592.11,"mail_bill_avg_minpayment_6mth":539.11,"email_billitem_max_single_freq_repay_6mth":11.0,"email_billitem_ratio_cnt_mthshopping_6mth":1.0,"mail_bill_min_lastpayment_6mth":0.0,"mail_card_min_limit_singlebank":4500.0,"email_billitem_cnt_mthwithdraw_6mth":0.0,"mail_bill_min_minpayment_6mth":220.6,"mail_card_min_cashlimit_singlebank":0.0,"mail_bill_avg_ratio_repay_6mth":1.12,"email_billitem_max_single_amount_6mth":49614.71,"max_overdue":1,"mail_bill_max_minpayment_6mth":1282.6,"email_billitem_max_single_freq_overdue_6mth":null,"mail_bill_max_ratio_nopay_6mth":0.33,"payamt_5":22914.0,"email_billitem_ratio_cnt_mthinstallment_6mth":0.17,"email_billitem_min_amount_shopping_6mth":2.0,"mail_card_sum_limit_singlebank":34000.0,"mail_bill_sum_bill_6mth":9.0,"email_billitem_sum_amount_overdue_6mth":null,"mail_bill_min_nopayment_6mth":-9952.5}'



headers = {
    'content-type': "application/json"
    }


response = requests.request("POST",url,data=payload,headers=headers)
print(response.text)

