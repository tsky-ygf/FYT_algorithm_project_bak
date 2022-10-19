import streamlit as st
import requests
from loguru import logger
import pandas as pd
from annotated_text import annotated_text
from annotated_text.util import get_annotated_html


def corrector_main(text, result):
    logger.debug(f'text:{text}, result: {result}')
    if result['success']:
        error_list = []
        if len(result['detail_info']) == 0:  # 说明没有错误
            markdown_corrected_pred = [text]
        else:
            error_text = ""
            markdown_corrected_pred = []
            start_index = 0
            for err in result['detail_info']:
                error_text += err[0] + "===>" + err[1] + "\n"
                markdown_corrected_pred.append(text[start_index:err[2]])
                markdown_corrected_pred.append((text[err[2]:err[3]], err[1], "#faa"))
                start_index = err[3]
            markdown_corrected_pred.append(text[err[3]:])
            #st.write(error_text)
            error_list.append(error_text)

        res_dict = {"原始文本": text, "纠错后的文本": result['corrected_pred'], "错别字": error_list}
        # import pdb
        # pdb.set_trace()
        my_df = pd.DataFrame.from_dict(res_dict)
        st.markdown('### 1.纠错后的文本')
        # import pdb
        # pdb.set_trace()
        annotated_text(*markdown_corrected_pred)
        st.write('-' * 100)
        st.markdown('### 2.表格结果')
        st.table(my_df)
    else:
        logger.error(result['msg'])
