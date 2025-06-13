#!/usr/bin/env python3

# 기본 시스템 및 ROS 관련 모듈
import os
import rclpy
from rclpy.node import Node
from dotenv import load_dotenv

# LangChain (OpenAI 연동 + 프롬프트 템플릿)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 서비스 타입 : string symptom → string medicine
from pharmacy_bot.srv import GetMedicineName


class SymptomMatcher(Node):
    """
    사용자의 증상을 받아 OpenAI API를 통해 적절한 약 이름을 추천하는 ROS 2 노드
    /get_medicine_name 서비스 서버를 통해 요청을 받고 응답을 반환한다
    """
    def __init__(self):
        super().__init__('symptom_matcher')

        # .env에서 openAI API 키 로딩
        package_path = os.getcwd()  # 설치 경로 또는 resource 위치로 바꿔도 됨
        load_dotenv(dotenv_path=os.path.join(package_path, ".env"))
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            self.get_logger().error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return

        # LangChain 기반 OpenAI 모델 설정
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.4,
            openai_api_key=openai_api_key
        )

        # 약 추천 프롬프트 설정 (추후 수정 필요)
        prompt_content = """
        당신은 약국에서 고객이 말하는 증상에 따라 적절한 일반의약품(OTC)을 추천해주는 전문가입니다.
        
        <목표>
        - 사용자의 증상 설명을 기반으로 일반의약품을 하나 추천하세요.
        - 반드시 약 이름(제품명)만 단답형으로 출력하세요.
        
        <주의사항>
        - 설명, 문장, 따옴표 없이 약 이름만 반환하세요.
        - 반드시 실제 존재하는 제품명만 제시하세요.
        
        <예시>
        - 입력: "기침이 나고 목이 아파요"
        출력: 판콜에이정
        
        - 입력: "소화가 안되고 배가 더부룩해요"
        출력: 베아제정
        
        <사용자 입력>
        "{user_input}"
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

        # 서비스 서버 설정 (/get_medicine_name)
        self.service = self.create_service(
            GetMedicineName, '/get_medicine_name', self.handle_symptom_request
        )

        self.get_logger().info("SymptomMatcher 서비스 노드 실행됨 (/get_medicine_name)")

    def handle_symptom_request(self, request, response):
        """
        증상 텍스트를 받아 약 이름을 생성해 응답하는 서비스 콜백 함수
        """
        symptom = request.symptom
        self.get_logger().info(f"증상 입력 받음: {symptom}")

        try:
            # LangChain LLMChain을 통해 약 추천 실행
            result = self.lang_chain.invoke({"user_input": symptom})
            recommended = result["text"].strip()

            self.get_logger().info(f"추천된 약: {recommended}")
            response.medicine = recommended
        except Exception as e:
            self.get_logger().error(f"OpenAI API 오류: {e}")
            response.medicine = "추천 실패"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SymptomMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

