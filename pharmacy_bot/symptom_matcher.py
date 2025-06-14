#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from dotenv import load_dotenv

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# ROS 2 서비스
from pharmacy_bot.srv import GetMedicineName


class SymptomMatcher(Node):
    def __init__(self):
        super().__init__('symptom_matcher')

        # 절대 경로로 .env 로드
        env_path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/.env")
        loaded = load_dotenv(dotenv_path=env_path)

        if not loaded:
            self.get_logger().warn(f".env 파일을 찾을 수 없습니다: {env_path}")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.get_logger().error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return

        # LangChain 모델 및 벡터 DB 초기화
        db_path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/resource/chroma_db")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        # ROS 2 서비스 생성
        self.service = self.create_service(GetMedicineName, '/get_medicine_name', self.handle_symptom_request)
        self.get_logger().info("SymptomMatcher 실행됨 (/get_medicine_name)")

    def translate_symptom(self, symptom: str) -> str:
        prompt = PromptTemplate(
            input_variables=["symptom"],
            template="""
            Translate the following Korean symptom description into English medical terms:
            "{symptom}"
            Respond only with the English translation.
            """
        )
        translated = self.llm.invoke(prompt.format(symptom=symptom))
        return translated.content.strip()

    def retrieve_context(self, translated_query: str) -> str:
        docs = self.retriever.get_relevant_documents(translated_query)
        return "\n".join([doc.page_content for doc in docs])

    def recommend_medicine(self, symptom: str, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["symptom", "context"],
            template="""
            당신은 반드시 다음 약물명(한글명)만 추천할 수 있습니다. 영어 이름으로 절대 출력하지 마세요.

            출력 가능한 약물명 리스트:
            ["모드콜", "콜대원", "하이펜", "타이레놀", "다제스", "락토프린", "포비돈", "미니온", "퓨어밴드", "Rohto C3 Cube"]

            약물 복용 안내:
            - 모드콜: 하루 3회, 1회 1정 복용. 졸림 유발 가능, 운전 주의.
            - 콜대원: 하루 3회, 1회 10ml 복용. 충분한 수분 섭취 권장.
            - 하이펜: 식후 1정 복용. 위장 장애 주의.
            - 타이레놀: 필요시 4~6시간 간격으로 1~2정 복용. 과다 복용 주의.
            - 다제스: 식사 직후 1~2정 복용.
            - 락토프린: 하루 2회, 공복 복용 권장.
            - 포비돈: 상처 부위에 적당량 도포. 사용 후 밀봉 보관.
            - 미니온: 통증 부위에 1일 1회 부착. 피부 트러블 시 제거.
            - 퓨어밴드: 상처 부위에 부착. 오염 시 교체.
            - Rohto C3 Cube: 1회 1~2방울, 1일 3~5회 점안. 사용 후 바로 뚜껑 닫기.

            <사용자 증상>
            {symptom}

            <관련 약물 판단 기준>
            {context}

            출력 형식:
            약 이름: [약 이름1, 약 이름2, ...
            복용 방법 및 주의 사항:
            - 약 이름1: (복용 방법)
            """
        )
        result = self.llm.invoke(prompt.format(symptom=symptom, context=context))
        return result.content.strip()

    def extract_first_drug_name(self, text: str) -> str:
        try:
            name_section = text.split("약 이름:")[1].split("]")[0]
            drug_list = name_section.replace("[", "").split(",")
            return drug_list[0].strip()
        except Exception as e:
            self.get_logger().error(f"약 이름 추출 실패: {e}")
            return "추천 실패"

    def handle_symptom_request(self, request, response):
        # symptom_query.txt 절대경로로 불러오기
        symptom_path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/resource/symptom_query.txt")
        try:
            with open(symptom_path, "r", encoding="utf-8") as f:
                symptom = f.read().strip()
            self.get_logger().info(f"증상 입력 파일에서 불러옴: {symptom}")
        except Exception as e:
            self.get_logger().error(f"symptom_query.txt 불러오기 실패: {e}")
            response.medicine = "추천 실패"
            return response

        try:
            translated = self.translate_symptom(symptom)
            context = self.retrieve_context(translated)
            result_text = self.recommend_medicine(symptom, context)
            self.get_logger().info(f"\n=== 약 추천 결과 ===\n{result_text}")

            drug_name = self.extract_first_drug_name(result_text)
            response.medicine = drug_name
        except Exception as e:
            self.get_logger().error(f"추천 중 오류 발생: {e}")
            response.medicine = "추천 실패"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SymptomMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

