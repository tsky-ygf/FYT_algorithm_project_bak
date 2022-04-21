# -*- coding: utf-8 -*-
import logging
import logging.handlers
import traceback
import json
import os
from flask import Flask
from flask import request
from flask import make_response, send_file
from flask import send_from_directory, render_template
from es_utils import *

app = Flask(__name__)
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler('./logs/service.log', when='D', interval=1, encoding='utf-8')
handler.setFormatter(formatter)
logger.addHandler(handler)


@app.route('/law/search', methods=["GET"])
def law_search():
    try:
        keywords = request.args.get('keywords', '')

        logging.info("=============================================================================")
        logging.info("keywords: %s" % (keywords))

        if keywords == '':
            return send_from_directory(os.path.abspath("./templates"), "law_content.html")
        else:
            result, count = law_content_search(keywords)
            logging.info("result size: %s" % (count))
            return json.dumps({"result": result, "download": 0, "size": count, "error_msg": "", "status": 0}, ensure_ascii=False)
    except Exception as e:
       logging.info(traceback.format_exc())
       return json.dumps({"result": [], "download": 0, "size": 0, "error_msg": traceback.format_exc(), "status": 1}, ensure_ascii=False)


@app.route('/law/search_whole', methods=["GET"])
def law_search_name():
    try:
        keywords = request.args.get('keywords', '')
        law_name = request.args.get('law_name', '')
        property = request.args.get('property', '')
        publish_date = request.args.get('publish_date', '')

        logging.info("=============================================================================")
        logging.info("keywords: %s; law_name: %s; property: %s; publish_date: %s" % (keywords, law_name, property, publish_date))

        if keywords == '':
            return send_from_directory(os.path.abspath("./templates"), "law_content.html")
        else:
            result, count = law_content_search_by_name(keywords, law_name, property, publish_date)
            logging.info("result size: %s" % (count))
            return render_template("law_full_content.html", result=result)
    except Exception as e:
       logging.info(traceback.format_exc())
       return json.dumps({"result": [], "download": 0, "size": 0, "error_msg": traceback.format_exc(), "status": 1}, ensure_ascii=False)


@app.route('/law/search_case', methods=["GET"])
def case_search():
    try:
        keywords = request.args.get('keywords', '')
        serial = request.args.get('serial', '')
        law_name = request.args.get('law_name', '')
        clause = request.args.get('clause', '')
        require_data = request.args.get('require_data', False)
        current_page = request.args.get('current_page', 0)

        logging.info("=============================================================================")
        logging.info("keywords: %s; serial: %s;" % (keywords, serial))

        if keywords == '' and (serial == '' or not require_data):
            return send_from_directory(os.path.abspath("./templates"), "case_content.html")
        elif keywords != '':
            result = case_content_search(keywords, current_page)
            logging.info("result size: %s" % (len(result)))
            return json.dumps({"result": result, "next_page": 1, "download": 1, "size": len(result), "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            result = case_content_search_by_serials(serial, law_name, clause)
            logging.info("result size: %s" % (len(result)))
            return json.dumps({"result": result, "next_page": 0, "download": 1, "size": len(result), "error_msg": "", "status": 0}, ensure_ascii=False)
    except Exception as e:
       logging.info(traceback.format_exc())
       return json.dumps({"result": [], "download": 0, "size": 0, "error_msg": traceback.format_exc(), "status": 1}, ensure_ascii=False)


@app.route('/law/download_case', methods=["GET", "POST"])
def download_case():
    try:
        logging.info(request.method)
        if request.method == 'GET':
            filename = request.args.get('filename', '')
            if filename != '':
                response = make_response(send_file('download/' + filename))
                response.headers["Content-disposition"] = 'attachment; filename=%s' % filename  # 如果不加上这行代码，导致下图的问题
                return response
        else:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            serials = json_data.get('serials', '')
            logging.info("=============================================================================")
            logging.info("serial: %s;" % (serials))

            if serials != '':
                filename = case_content_download_by_serials(serials)
                logging.info(filename + " generate succeed")
                return json.dumps({"result": filename, "status": 1}, ensure_ascii=False)

        return json.dumps({"error_msg": "No data to download", "status": 1}, ensure_ascii=False)
    except Exception as e:
       logging.info(traceback.format_exc())
       return json.dumps({"error_msg": traceback.format_exc(), "status": 1}, ensure_ascii=False)


if __name__=='__main__':
    app.run(host="0.0.0.0", port=5010, debug=True, use_reloader=True)
