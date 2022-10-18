import streamlit as st
import requests
from loguru import logger
import pandas as pd

def corrector_main():
    tab1 = st.tabs(["纠错任务"])

    with tab1:
        text = st.text_input("请输入您的问题", value="真麻烦你了。希望你们好好的跳无", key="样例")
        result = requests.post("http://0.0.0.0:6598/get_corrected_contract_result",
                            json={"text": text}).json()

        if result['success']:
            error_list = []
            if len(result['detail_info']) == 0:#说明没有错误
                pass
            else:
                error_text = ""
                for err in result['detail_info']:
                    error_text += err[0] + "===>" + err[1] + "\n"
                st.write(error_text)
                error_list.append(error_text)
            res_dict = {"原始文本": text, "纠错后的文本": result['corrected_pred'], "错别字": error_list}
            my_df = pd.DataFrame.from_dict(res_dict)
            st.table(my_df)
        else:
            logger.error(result['msg'])
