import asyncio
import aiohttp

url = 'https://cloudrun-inference-5dfmxxj4uq-du.a.run.app/'
request_model = "mobilenet"
try_number = 1000


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(try_number):
            task = asyncio.ensure_future(fetch(session, (url+request_model)))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        for response in responses:
            print(response)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())