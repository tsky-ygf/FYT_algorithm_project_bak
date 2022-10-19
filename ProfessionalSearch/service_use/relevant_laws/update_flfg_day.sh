cd /home/fyt/huangyulin/project/fyt
source /home/fyt/miniconda3/bin/activate hyl_search_py39
nohup python -u ProfessionalSearch/service_use/relevant_laws/process_law_to_es.py > process_law_to_es.log 2>&1 &