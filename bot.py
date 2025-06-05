import hashlib, time, random, aiohttp, asyncio, numpy as np, os
from typing import List, Optional, Tuple
import concurrent.futures
import shareithub
from shareithub import shareithub

def generate_matrix(seed: int, size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float64)
    current_seed = seed
    a, b = 0x4b72e682d, 0x2675dcd22
    for i in range(size):
        for j in range(size):
            value = (a * current_seed + b) % 1000
            matrix[i][j] = float(value)
            current_seed = value
    return matrix

def multiply_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b

def flatten_matrix(matrix: np.ndarray) -> str:
    return ''.join(f"{x:.0f}" for x in matrix.flat)

async def compute_hash_mod(matrix: np.ndarray, mod: int = 10**7) -> int:
    flat_str = flatten_matrix(matrix)
    sha256 = hashlib.sha256(flat_str.encode()).hexdigest()
    return int(int(sha256, 16) % mod)

async def fetch_task(session: aiohttp.ClientSession, token: str) -> Tuple[dict, bool]:
    headers = {"Content-Type": "application/json", "token": token}
    try:
        async with session.post("https://nebulai.network/open_compute/finish/task", json={}, headers=headers, timeout=10) as resp:
            data = await resp.json()
            print(f"[🔍] Full fetch response for {token[:8]}: {data}")  # <<< Tambahan ini
            if data.get("code") == 0:
                print(f"[📥] Task OK for {token[:8]} (size: {data['data']['matrix_size']})")
                return data['data'], True
            return None, False
    except Exception as e:
        print(f"[⚠️] Fetch error for {token[:8]}: {str(e)}")
        return None, False


async def submit_results(session: aiohttp.ClientSession, token: str, r1: float, r2: float, task_id: str) -> bool:
    headers = {"Content-Type": "application/json", "token": token}
    payload = {"result_1": f"{r1:.10f}", "result_2": f"{r2:.10f}", "task_id": task_id}
    try:
        async with session.post("https://nebulai.network/open_compute/finish/task", json=payload, headers=headers, timeout=10) as resp:
            data = await resp.json()
            print(f"[📤] Full submit response for {token[:8]}: {data}")  # <<< Tambahan ini
            if data.get("code") == 0 and data.get("data", {}).get("calc_status", False):
                print(f"[✅] Accepted for {token[:8]}")
                return True
            print(f"[❌] Rejected for {token[:8]}: {data}")
            return False
    except Exception as e:
        print(f"[⚠️] Submit error for {token[:8]}: {str(e)}")
        return False


async def process_task(token: str, task_data: dict) -> Optional[Tuple[float, float]]:
    seed1, seed2, size = task_data["seed1"], task_data["seed2"], task_data["matrix_size"]
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t0 = time.time() * 1000
            A_future = executor.submit(generate_matrix, seed1, size)
            B_future = executor.submit(generate_matrix, seed2, size)
            A, B = await asyncio.gather(
                asyncio.wrap_future(A_future),
                asyncio.wrap_future(B_future)
            )
        C = multiply_matrices(A, B)
        f = await compute_hash_mod(C)
        t1 = time.time() * 1000
        result_1 = t0 / f
        result_2 = f / (t1 - t0) if (t1 - t0) != 0 else 0
        return result_1, result_2
    except Exception as e:
        print(f"[❌] Compute error: {str(e)}")
        return None

async def worker_loop(token: str):
    async with aiohttp.ClientSession() as session:
        while True:
            task_data, success = await fetch_task(session, token)
            if not success:
                await asyncio.sleep(2)
                continue
            results = await process_task(token, task_data)
            if not results:
                await asyncio.sleep(1)
                continue
            submitted = await submit_results(session, token, results[0], results[1], task_data["task_id"])
            await asyncio.sleep(0.5 if submitted else 3)

async def main():
    if not os.path.exists("tokens.txt"):
        print("tokens.txt not found!")
        return
    with open("tokens.txt") as f:
        tokens = [line.strip() for line in f if line.strip()]
    if not tokens:
        print("No tokens in tokens.txt!")
        return
    await asyncio.gather(*(worker_loop(token) for token in tokens))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[⛔] Stopped by user")
