import requests

def simple_llm_request(prompt):
    req_json = {
        "stream": False,
        "n_predict": 500,
        "temperature": 0.7,
        "stop": [
            "</s>",
        ],
        "top_k": 20,
        "top_p": 0.75,
        "tfs_z": 1,
        "typical_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "mirostat": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "grammar": "",
        "n_probs": 0,
        "prompt": f"<s>[INST] {prompt} [/INST]"
    }

    url = "http://localhost:8080/completion"
    res = requests.post(url, json=req_json)

    result = res.json()["content"]

    return result
    

if __name__ == "__main__":
    base_template = """
    너는 버킷리스트를 이루는 것을 도와주는 역할을 맡은 멘토야. 큰 목표가 주어지면 너는 이 목표를 이루어나갈 수 있도록 큰 목표를 작은 목표들로 나눈 5단계를 제시해줘. 나의 목표는 "{bucketlist}"이고, 이 목표를 이루어나가기 위한 5가지 작은 단계를 자세한 작은 목표와 함께 설명 없이 제시해줘.
    """
    prompt = base_template.format(bucketlist="수능 만점 받기")
    print(simple_llm_request(prompt))