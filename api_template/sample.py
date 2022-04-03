from pydantic import BaseModel


class Sample(BaseModel):
    Age: int = 19
    BusinessTravel: str = "Non-Travel"
    DailyRate: int = 10000
    Department: str = "Sales"
    DistanceFromHome: int = 100
    Education: int = 1
    EducationField: str = "Human Resources"
    EmployeeCount: int = 1
    EmployeeNumber: int = 1
    EnvironmentSatisfaction: int = 3
    Gender: str = "Female"
    HourlyRate: int = 10000
    JobInvolvement: int = 3
    JobLevel: int = 3
    JobRole: str = "Human Resources"
    JobSatisfaction: int = 3
    MaritalStatus: str = "Married"
    MonthlyIncome: int = 10000
    MonthlyRate: int = 10000
    NumCompaniesWorked: int = 1
    Over18: str = "Yes"
    OverTime: str = "No"
    PercentSalaryHike: int = 1
    PerformanceRating: int = 3
    RelationshipSatisfaction: int = 3
    StandardHours: int = 8
    StockOptionLevel: int = 1
    TotalWorkingYears: int = 1
    TrainingTimesLastYear: int = 1
    WorkLifeBalance: int = 3
    YearsAtCompany: int = 1
    YearsInCurrentRole: int = 3
    YearsSinceLastPromotion: int = 3
    YearsWithCurrManager: int = 3