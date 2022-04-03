from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/whoami")
def whoami() -> str:
    # TODO
    isim = "Enes"
    soyisim = "Can"
    mail = "mail@mail.com"

    person_card = {
        "isim": isim,
        "soyisim": soyisim,
        "mail": mail
    }

    return person_card


if __name__ == "__main__":
    uvicorn.run("deneme:app", host="127.0.0.1", port=5000, log_level="info")
