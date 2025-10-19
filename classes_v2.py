#################################################################################
# Chat2VIS 支持函数
# https://chat2vis.streamlit.app/
# Paula Maddigan
#################################################################################

import os
from openai import OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

def run_request(question_to_ask, model_type, key, alt_key):
    # 根据模型类型调用不同的 API 获取 Python 代码脚本
    if model_type == "deepseek-chat":
        # 调用 DeepSeek API
        client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com"
        )
        task = "Generate Python Code Script. The script should only include code, no comments."
        response = client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": task},
                {"role": "user", "content": question_to_ask}
            ],
            stream=False
        )
        llm_response = response.choices[0].message.content
    elif model_type == "gpt-4" or model_type == "gpt-3.5-turbo" :
        # 调用 OpenAI ChatCompletion API
        task = "Generate Python Code Script."
        if model_type == "gpt-4":
            # 保证 GPT-4 只返回代码，不包含注释
            task = task + " The script should only include code, no comments."
        openai.api_key = key
        response = openai.ChatCompletion.create(model=model_type,
            messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
        llm_response = response["choices"][0]["message"]["content"]
    elif model_type == "text-davinci-003" or model_type == "gpt-3.5-turbo-instruct":
        # 调用 OpenAI Completion API
        openai.api_key = key
        response = openai.Completion.create(engine=model_type,prompt=question_to_ask,temperature=0,max_tokens=500,
                    top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,stop=["st.pyplot(fig)"])
        llm_response = response["choices"][0]["text"] 
    else:
        # 调用 Hugging Face 模型
        llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id="codellama/" + model_type, model_kwargs={"temperature":0.1, "max_new_tokens":500})
        llm_prompt = PromptTemplate.from_template(question_to_ask)
        llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
        llm_response = llm_chain.predict()
    # 格式化模型返回结果
    llm_response = format_response(llm_response)
    return llm_response

def format_response(res):
    # 如果答案中包含 read_csv，则移除相关行
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # read_csv 行是第一行，无需保留前面的内容
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # read_csv 是最后一行
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc, primer_code, question, model_type):
    # 填充模型特定的指令
    instructions = ""
    primer_desc = primer_desc.format(instructions)
    # 将问题放在描述 primer 的结尾，并加上代码 primer
    return '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset, df_name):
    # primer 函数：根据数据集和名称生成描述和代码片段
    # 包含所有列名，若某列唯一值少于 20 且为类别型，则补充所有类别值
    # 并设置水平网格线和标签
    primer_desc = "使用名为 df 的 dataframe，来源 data_file.csv，包含列 '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'。"
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i] == "O":
            primer_desc = primer_desc + "\n列 '" + i + "' 的类别值有 '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'。"
        elif df_dataset.dtypes[i] == "int64" or df_dataset.dtypes[i] == "float64":
            primer_desc = primer_desc + "\n列 '" + i + "' 类型为 " + str(df_dataset.dtypes[i]) + "，包含数值型数据。"
    primer_desc = primer_desc + "\n请合理标注 x 和 y 轴。"
    primer_desc = primer_desc + "\n添加标题，fig suptitle 设为空。"
    primer_desc = primer_desc + "{}" # 预留补充指令
    primer_desc = primer_desc + "\n请使用 Python 3.11.13，基于 dataframe df 编写绘图脚本："
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc, pimer_code