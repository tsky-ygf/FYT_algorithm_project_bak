import streamlit as st
import requests
from loguru import logger
import pandas as pd


def corrector_main(text, result):
    logger.debug(f'text:{text}, result: {result}')
    if result['success']:
        error_list = []
        if len(result['detail_info']) == 0:  # 说明没有错误
            pass
        else:
            error_text = ""
            for err in result['detail_info']:
                error_text += err[0] + "===>" + err[1] + "\n"
            #st.write(error_text)
            error_list.append(error_text)
        res_dict = {"原始文本": text, "纠错后的文本": result['corrected_pred'], "错别字": error_list}
        my_df = pd.DataFrame.from_dict(res_dict)
        st.table(my_df)
    else:
        logger.error(result['msg'])
