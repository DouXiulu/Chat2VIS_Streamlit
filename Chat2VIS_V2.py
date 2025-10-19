#################################################################################
# Chat2VIS 主程序
# https://chat2vis.streamlit.app/
# Paula Maddigan
#################################################################################

import pandas as pd
import os
from openai import OpenAI
import streamlit as st
#import streamlit_nested_layout
from classes_v2 import get_primer,format_question,run_request
import warnings
warnings.filterwarnings("ignore")
# st.set_option('deprecation.showPyplotGlobalUse', False)  # 此选项在新版本中已弃用
st.set_page_config(page_icon="chat2vis.png",layout="wide",page_title="Chat2VIS")


st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
            Chat2VIS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations using Natural Language \
            with DeepSeek</h2>", unsafe_allow_html=True)

st.sidebar.markdown('</a> Developed by Paula Maddigan <a style="text-align: center;padding-top: 0rem;" href="mailto: i.build.apps.4.u@gmail.com">:email:', unsafe_allow_html=True)


available_models = {"DeepSeek-Chat": "deepseek-chat"}

## 用于保存数据集的列表
if "datasets" not in st.session_state:
    datasets = {}
    # 预加载数据集
    datasets["Movies"] = pd.read_csv("movies.csv")
    datasets["Housing"] = pd.read_csv("housing.csv")
    datasets["Cars"] = pd.read_csv("cars.csv")
    datasets["Colleges"] = pd.read_csv("colleges.csv")
    datasets["Customers & Products"] = pd.read_csv("customers_and_products_contacts.csv")
    datasets["Department Store"] = pd.read_csv("department_store.csv")
    datasets["Energy Production"] = pd.read_csv("energy_production.csv")
    st.session_state["datasets"] = datasets
else:
    # 使用已加载的数据集列表
    datasets = st.session_state["datasets"]

key_col1,key_col2 = st.columns(2)
deepseek_key = key_col1.text_input(label = ":key: DeepSeek API Key:", help="Required for DeepSeek-Chat.",type="password")
key_col2.empty()  # 保留布局但不再使用HuggingFace Key

with st.sidebar:
    # 首先选择数据集，加载后填充选项
    dataset_container = st.empty()

    # 添加上传数据集功能
    try:
        uploaded_file = st.file_uploader(":computer: 加载 CSV 文件:", type="csv")
        index_no = 0
        if uploaded_file:
            # 读取数据并加入可选数据集列表，自动生成名称
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            # 默认选中新上传的数据集
            index_no = len(datasets) - 1
    except Exception as e:
        st.error("文件加载失败，请选择有效的 CSV 文件。")
        print("文件加载失败。\n" + str(e))
    # 数据集选择单选框
    chosen_dataset = dataset_container.radio(":bar_chart: 选择数据:", datasets.keys(), index=index_no)

    # 模型选择复选框
    st.write(":brain: 选择模型:")
    # 记录每个模型是否被选中
    use_model = {}
    for model_desc, model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label, value=True, key=key)
 
## 可视化问题输入框
question = st.text_area(":eyes: 你想可视化什么内容？", height=10)
go_btn = st.button("开始生成...")

# 统计已选中的模型列表
selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(selected_models)

# 执行聊天机器人请求
if go_btn and model_count > 0:
    api_keys_entered = True
    # 检查 API 密钥是否输入
    if "DeepSeek-Chat" in selected_models:
        if not deepseek_key:
            st.error("请输入有效的 DeepSeek API 密钥。")
            api_keys_entered = False
    if api_keys_entered:
        # 根据模型数量创建绘图区
        plots = st.columns(model_count)
        # 获取当前数据集的 primer
        primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
        # 按模型依次生成并展示结果
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    # 格式化问题
                    question_to_ask = format_question(primer1, primer2, question, model_type)
                    # 执行请求
                    answer = ""
                    answer = run_request(question_to_ask, available_models[model_type], key=deepseek_key, alt_key="")
                    # answer 为完整 Python 脚本，需加 primer2 头部
                    answer = primer2 + answer
                    print("模型: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))
                except Exception as e:
                    st.error("模型生成的代码有误，无法执行。(" + str(e) + ")")

## 以标签页形式展示所有数据集
tab_list = st.tabs(datasets.keys())

# 每个标签页加载对应数据集
for dataset_num, tab in enumerate(tab_list):
    with tab:
        # 通过索引获取数据集名称
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)

## 页脚，注明数据集来源
footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class=\"footer\">
<p> <a style='display: block; text-align: center;'> 数据集来源：NL4DV、nvBench、ADVISor </a></p></div>"""
st.caption("数据集来源：NL4DV、nvBench、ADVISor")

## 隐藏 Streamlit 菜单和默认页脚
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
