from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SleepStressInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "age": 29,
                "occupation": "Nurse",
                "sleep_duration": 6.5,
                "quality_of_sleep": 6,
                "physical_activity_level": 45,
                "bmi_category": "Normal",
                "systolic": 120,
                "diastolic": 80,
                "heart_rate": 72,
                "daily_steps": 6000,
                "sleep_disorder": "None",
            }
        }
    )

    gender: Literal["Male", "Female"]
    age: int = Field(..., ge=10, le=100)
    occupation: str = Field(..., min_length=2, max_length=60)
    sleep_duration: float = Field(..., ge=0, le=24, description="Hours per night")
    quality_of_sleep: int = Field(..., ge=1, le=10)
    physical_activity_level: int = Field(..., ge=0, le=600, description="Minutes per day")
    bmi_category: Literal["Normal", "Normal Weight", "Overweight", "Obese"]
    systolic: int = Field(..., ge=70, le=250)
    diastolic: int = Field(..., ge=40, le=150)
    heart_rate: int = Field(..., ge=30, le=220)
    daily_steps: int = Field(..., ge=0, le=60000)
    sleep_disorder: Literal["None", "Insomnia", "Sleep Apnea"]

    @model_validator(mode="after")
    def check_blood_pressure(self):
        if self.diastolic >= self.systolic:
            raise ValueError("diastolic must be lower than systolic")
        return self


class SleepStressOutput(BaseModel):
    predicted_stress_level: float = Field(..., description="Estimated score on a 1-10 scale")
    category: Literal["Low", "Moderate", "High"]
    disclaimer: str