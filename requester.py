import asyncio
import aiohttp
import csv
import datetime
import time

current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
url = 'https://cloudrun-inference-5dfmxxj4uq-du.a.run.app/'
request_model = "mobilenet"
try_number = 10
csv_file_name = f"./{current_time}.csv"

script_start_time = time.monotonic()

data = [["inference_time", "elapsed_time", "min_inference_time", "max_inference_time", "min_comsumed_time", "max_elapsed_time", "average_inference_time", "avearage_elapsed_time", "script_time"]]

elapsed_time_data = []

def save_csv():
    with open(csv_file_name, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def add_multiple_time_units():
    sum_inference_time = 0
    sum_elapsed_time = 0
    min_inference_time = 1
    max_inference_time = 0
    min_elapsed_time = 1
    max_elapsed_time = 0
    for i in range(1,len(data)):
      inference_time = data[i][0]
      elapsed_time = data[i][1]
      sum_inference_time += float(inference_time)
      sum_elapsed_time += float(elapsed_time)
      
      max_elapsed_time = float(max_elapsed_time) if max_elapsed_time > float(elapsed_time) else float(elapsed_time)
      min_elapsed_time = float(min_elapsed_time) if min_elapsed_time < float(elapsed_time) else float(elapsed_time)
      max_inference_time = float(max_inference_time) if max_inference_time > float(inference_time) else float(inference_time)
      min_inference_time = float(min_inference_time) if min_inference_time < float(inference_time) else float(inference_time)

    average_inference_time = sum_inference_time / (len(data)-1)
    average_elapsed_time = sum_elapsed_time / (len(data)-1)
    data[1].append(f"{min_inference_time:.17f}")
    data[1].append(f"{max_inference_time:.17f}")
    data[1].append(f"{min_elapsed_time:.17f}")
    data[1].append(f"{max_elapsed_time:.17f}")
    data[1].append(f"{average_inference_time:.17f}")
    data[1].append(f"{average_elapsed_time:.17f}")


async def fetch(session, url):
    start_time = time.monotonic()
    async with session.get(url) as response:
        end_time = time.monotonic()
        return await response.text(), end_time - start_time


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(try_number):
            task = asyncio.ensure_future(fetch(session, (url+request_model)))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        count = 1
        for response in responses:
            lines = response[0].split('\n')
            splits = lines[1].split()
            inference_time = splits[2]
            new_line = [inference_time]
            data.append(new_line)
            # elapsed_time_data.append(response[1])
            data[count].append(f"{float(response[1]):.17f}")
            count += 1
    add_multiple_time_units()
    script_end_time = time.monotonic() - script_start_time
    data[1].append(script_end_time)
    save_csv()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
