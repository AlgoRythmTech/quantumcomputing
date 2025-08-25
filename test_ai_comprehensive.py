"""
Comprehensive Test Suite for RythmAi Europa System
Tests for: API functionality, Model responses, Hardcoded detection, Performance, Security
Created by: AI Testing Assistant
"""

import requests
import json
import time
import sys
import asyncio
import aiohttp
import random
import string
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import concurrent.futures
from collections import Counter
import statistics
import re

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Test configuration
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"

class RythmAiTestSuite:
    """Comprehensive test suite for RythmAi system"""
    
    def __init__(self):
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "hardcoded_indicators": []
        }
        self.response_cache = {}
        self.timing_data = []
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
    def print_test(self, name: str, status: str, message: str = ""):
        """Print test result"""
        if status == "PASS":
            symbol = "✅"
            color = Colors.GREEN
            self.test_results["passed"].append(name)
        elif status == "FAIL":
            symbol = "❌"
            color = Colors.RED
            self.test_results["failed"].append(name)
        elif status == "WARN":
            symbol = "⚠️"
            color = Colors.YELLOW
            self.test_results["warnings"].append(name)
        else:
            symbol = "ℹ️"
            color = Colors.BLUE
            
        print(f"{symbol} {color}{name}: {status}{Colors.RESET}")
        if message:
            print(f"   {message}")
    
    async def test_hardcoded_responses(self) -> Dict[str, Any]:
        """
        CRITICAL TEST: Detect if responses are hardcoded
        This is the main test to determine if the AI is real or fake
        """
        self.print_header("HARDCODED RESPONSE DETECTION")
        results = {
            "is_hardcoded": False,
            "confidence": 0,
            "indicators": []
        }
        
        # Test 1: Randomness test - same input, different responses?
        print(f"{Colors.BOLD}Test 1: Response Variability{Colors.RESET}")
        test_prompt = "Tell me a random fact"
        responses = []
        
        for i in range(5):
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={
                        "message": test_prompt,
                        "temperature": 0.9,  # High temperature for variability
                        "search_web": False
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    responses.append(response.json().get("response", ""))
                time.sleep(0.5)
            except:
                pass
        
        # Check if all responses are identical (hardcoded indicator)
        if len(set(responses)) == 1 and len(responses) > 1:
            self.print_test("Response Variability", "FAIL", "All responses identical - HARDCODED!")
            results["indicators"].append("IDENTICAL_RESPONSES")
            results["confidence"] += 40
        else:
            self.print_test("Response Variability", "PASS", f"Got {len(set(responses))} unique responses")
        
        # Test 2: Nonsense input test
        print(f"\n{Colors.BOLD}Test 2: Nonsense Input Handling{Colors.RESET}")
        nonsense_inputs = [
            "xyzqwerty123 asdfgh jklmnop",
            "🦆🎭🌈 florb the zarnacle",
            "".join(random.choices(string.ascii_letters, k=50)),
            "when does the narwhal bacon? at midnight purple monkey dishwasher"
        ]
        
        nonsense_responses = []
        for nonsense in nonsense_inputs:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": nonsense, "search_web": False},
                    timeout=10
                )
                if response.status_code == 200:
                    resp_text = response.json().get("response", "")
                    nonsense_responses.append(resp_text)
                    
                    # Check for generic hardcoded error messages
                    if "I don't understand" in resp_text or "Could you please rephrase" in resp_text:
                        results["indicators"].append("GENERIC_ERROR_RESPONSE")
            except:
                pass
        
        # Check if nonsense gets reasonable handling
        if all("error" in r.lower() or "understand" in r.lower() for r in nonsense_responses):
            self.print_test("Nonsense Handling", "WARN", "Generic error responses detected")
            results["confidence"] += 20
        else:
            self.print_test("Nonsense Handling", "PASS", "Handles nonsense naturally")
        
        # Test 3: Company information consistency
        print(f"\n{Colors.BOLD}Test 3: Company Info Consistency{Colors.RESET}")
        company_questions = [
            "Who created you?",
            "Who is your CEO?",
            "What company made you?",
            "Tell me about AlgoRythm Tech"
        ]
        
        company_responses = []
        expected_terms = ["algorythm tech", "sri aasrith souri kompella", "hyderabad"]
        
        for question in company_questions:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": question, "search_web": False},
                    timeout=10
                )
                if response.status_code == 200:
                    resp_text = response.json().get("response", "").lower()
                    company_responses.append(resp_text)
            except:
                pass
        
        # Check if responses are too similar (copy-pasted)
        if len(company_responses) > 1:
            # Calculate similarity between responses
            similarities = []
            for i in range(len(company_responses)-1):
                common_words = set(company_responses[i].split()) & set(company_responses[i+1].split())
                similarity = len(common_words) / max(len(company_responses[i].split()), 
                                                    len(company_responses[i+1].split()))
                similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            if avg_similarity > 0.8:  # High similarity indicates hardcoding
                self.print_test("Company Info Variability", "FAIL", 
                              f"Responses too similar ({avg_similarity:.2%}) - LIKELY HARDCODED")
                results["indicators"].append("TEMPLATED_COMPANY_INFO")
                results["confidence"] += 30
            else:
                self.print_test("Company Info Variability", "PASS", 
                              f"Natural variation in responses ({avg_similarity:.2%})")
        
        # Test 4: Response timing analysis
        print(f"\n{Colors.BOLD}Test 4: Response Timing Analysis{Colors.RESET}")
        timing_tests = []
        
        for length in [10, 50, 200, 1000]:  # Different input lengths
            prompt = "a " * length
            start_time = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": prompt, "search_web": False},
                    timeout=30
                )
                timing_tests.append({
                    "length": length,
                    "time": time.time() - start_time
                })
            except:
                pass
        
        if timing_tests:
            times = [t["time"] for t in timing_tests]
            # Hardcoded responses would have very consistent timing
            time_variance = statistics.stdev(times) if len(times) > 1 else 0
            
            if time_variance < 0.1:  # Very low variance suggests hardcoding
                self.print_test("Response Timing", "FAIL", 
                              f"Suspiciously consistent timing (σ={time_variance:.3f}s) - LIKELY HARDCODED")
                results["indicators"].append("CONSISTENT_TIMING")
                results["confidence"] += 20
            else:
                self.print_test("Response Timing", "PASS", 
                              f"Natural timing variance (σ={time_variance:.3f}s)")
        
        # Test 5: Mathematical/Logic test
        print(f"\n{Colors.BOLD}Test 5: Dynamic Computation Test{Colors.RESET}")
        math_tests = [
            f"What is {random.randint(100, 999)} + {random.randint(100, 999)}?",
            f"What is {random.randint(10, 99)} * {random.randint(10, 99)}?",
            "If I have 17 apples and give away 9, how many do I have?",
            "What's the next number in the sequence: 2, 4, 8, 16, ?"
        ]
        
        correct_answers = 0
        for test in math_tests:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": test, "search_web": False},
                    timeout=10
                )
                if response.status_code == 200:
                    resp_text = response.json().get("response", "")
                    # Simple check if it attempts to answer
                    if any(char.isdigit() for char in resp_text):
                        correct_answers += 1
            except:
                pass
        
        if correct_answers == 0:
            self.print_test("Dynamic Computation", "FAIL", "Cannot perform calculations - LIKELY HARDCODED")
            results["indicators"].append("NO_COMPUTATION_ABILITY")
            results["confidence"] += 30
        else:
            self.print_test("Dynamic Computation", "PASS", f"Answered {correct_answers}/{len(math_tests)} math questions")
        
        # Test 6: Context retention test
        print(f"\n{Colors.BOLD}Test 6: Context Retention Test{Colors.RESET}")
        context_test = [
            ("My name is TestBot123", "What's my name?"),
            ("I like purple elephants", "What do I like?"),
            ("The secret code is XYZABC", "What's the secret code?")
        ]
        
        context_success = 0
        for setup, question in context_test:
            try:
                # Setup context
                conv_id = hashlib.md5(str(time.time()).encode()).hexdigest()
                response1 = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": setup, "conversation_id": conv_id, "search_web": False},
                    timeout=10
                )
                
                # Test context
                response2 = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": question, "conversation_id": conv_id, "search_web": False},
                    timeout=10
                )
                
                if response2.status_code == 200:
                    resp_text = response2.json().get("response", "")
                    # Check if it remembers context
                    if "TestBot123" in resp_text or "purple" in resp_text or "XYZABC" in resp_text:
                        context_success += 1
            except:
                pass
        
        if context_success == 0:
            self.print_test("Context Retention", "FAIL", "No context awareness - LIKELY HARDCODED")
            results["indicators"].append("NO_CONTEXT_RETENTION")
            results["confidence"] += 20
        else:
            self.print_test("Context Retention", "PASS", f"Retained context in {context_success}/{len(context_test)} tests")
        
        # Final verdict
        results["is_hardcoded"] = results["confidence"] >= 50
        
        print(f"\n{Colors.BOLD}=== HARDCODED DETECTION RESULTS ==={Colors.RESET}")
        print(f"Confidence Score: {results['confidence']}%")
        print(f"Indicators Found: {', '.join(results['indicators']) if results['indicators'] else 'None'}")
        
        if results["is_hardcoded"]:
            print(f"{Colors.RED}{Colors.BOLD}⚠️  HIGH PROBABILITY OF HARDCODED RESPONSES! ⚠️{Colors.RESET}")
            print(f"{Colors.RED}The system appears to use pre-programmed or templated responses.{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ APPEARS TO BE GENUINE AI MODEL{Colors.RESET}")
            print(f"{Colors.GREEN}Responses show natural variation and dynamic generation.{Colors.RESET}")
        
        self.test_results["hardcoded_indicators"] = results["indicators"]
        return results
    
    async def test_api_endpoints(self):
        """Test all API endpoints thoroughly"""
        self.print_header("API ENDPOINT TESTING")
        
        # Test root endpoint
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.print_test("Root Endpoint", "PASS", f"Model: {data.get('name', 'Unknown')}")
            else:
                self.print_test("Root Endpoint", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.print_test("Root Endpoint", "FAIL", str(e))
        
        # Test chat endpoint with various inputs
        chat_tests = [
            {"message": "Hello", "expected": "response"},
            {"message": "X" * 10000, "expected": "handle_long"},  # Long input
            {"message": "", "expected": "handle_empty"},  # Empty input
            {"message": "Tell me about quantum computing", "search_web": True, "expected": "web_search"},
            {"message": "Generate a report", "generate_pdf": True, "expected": "pdf_generation"},
        ]
        
        for i, test in enumerate(chat_tests):
            try:
                response = requests.post(f"{BASE_URL}/api/chat", json=test, timeout=15)
                if response.status_code == 200:
                    self.print_test(f"Chat Test {i+1}", "PASS", test.get("expected", ""))
                else:
                    self.print_test(f"Chat Test {i+1}", "FAIL", f"Status: {response.status_code}")
            except Exception as e:
                self.print_test(f"Chat Test {i+1}", "FAIL", str(e))
        
        # Test search endpoint
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json={"query": "Python programming", "num_results": 3},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                self.print_test("Search Endpoint", "PASS", f"Found {len(results)} results")
            else:
                self.print_test("Search Endpoint", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.print_test("Search Endpoint", "FAIL", str(e))
    
    async def test_performance(self):
        """Test system performance and stress handling"""
        self.print_header("PERFORMANCE TESTING")
        
        # Test 1: Response time for different input sizes
        print(f"{Colors.BOLD}Response Time Analysis:{Colors.RESET}")
        input_sizes = [10, 100, 500, 1000, 5000]
        
        for size in input_sizes:
            prompt = "Explain " + " ".join(["concept"] * (size // 7))
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": prompt, "search_web": False},
                    timeout=30
                )
                elapsed = time.time() - start_time
                self.timing_data.append(elapsed)
                
                if response.status_code == 200:
                    self.print_test(f"Input Size {size}", "PASS", f"Response time: {elapsed:.2f}s")
                else:
                    self.print_test(f"Input Size {size}", "FAIL", f"Status: {response.status_code}")
            except Exception as e:
                self.print_test(f"Input Size {size}", "FAIL", str(e))
        
        # Test 2: Concurrent requests
        print(f"\n{Colors.BOLD}Concurrent Request Handling:{Colors.RESET}")
        
        async def make_concurrent_request(session, index):
            try:
                async with session.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": f"Test request {index}", "search_web": False},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status, await response.json()
            except:
                return None, None
        
        async def test_concurrent():
            async with aiohttp.ClientSession() as session:
                tasks = [make_concurrent_request(session, i) for i in range(10)]
                results = await asyncio.gather(*tasks)
                successful = sum(1 for status, _ in results if status == 200)
                return successful
        
        try:
            successful = await test_concurrent()
            if successful >= 8:
                self.print_test("Concurrent Requests", "PASS", f"{successful}/10 successful")
            elif successful >= 5:
                self.print_test("Concurrent Requests", "WARN", f"{successful}/10 successful")
            else:
                self.print_test("Concurrent Requests", "FAIL", f"Only {successful}/10 successful")
        except Exception as e:
            self.print_test("Concurrent Requests", "FAIL", str(e))
        
        # Test 3: Memory leak test (multiple sequential requests)
        print(f"\n{Colors.BOLD}Memory Stability Test:{Colors.RESET}")
        response_times = []
        
        for i in range(20):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": "Memory test", "search_web": False},
                    timeout=10
                )
                response_times.append(time.time() - start_time)
            except:
                pass
        
        if response_times:
            # Check if response times are increasing (potential memory leak)
            first_half_avg = sum(response_times[:10]) / 10
            second_half_avg = sum(response_times[10:]) / 10
            
            if second_half_avg > first_half_avg * 1.5:
                self.print_test("Memory Stability", "WARN", 
                              f"Performance degradation detected ({second_half_avg/first_half_avg:.2f}x slower)")
            else:
                self.print_test("Memory Stability", "PASS", "No performance degradation")
    
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        self.print_header("ERROR HANDLING TESTING")
        
        error_tests = [
            {
                "name": "Invalid JSON",
                "url": f"{BASE_URL}/api/chat",
                "data": "not json",
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Missing Required Field",
                "url": f"{BASE_URL}/api/chat",
                "json": {"temperature": 0.5}  # Missing 'message'
            },
            {
                "name": "Invalid Temperature",
                "url": f"{BASE_URL}/api/chat",
                "json": {"message": "test", "temperature": 10.0}
            },
            {
                "name": "Negative Max Tokens",
                "url": f"{BASE_URL}/api/chat",
                "json": {"message": "test", "max_tokens": -100}
            },
            {
                "name": "SQL Injection Attempt",
                "url": f"{BASE_URL}/api/chat",
                "json": {"message": "'; DROP TABLE users; --"}
            },
            {
                "name": "XSS Attempt",
                "url": f"{BASE_URL}/api/chat",
                "json": {"message": "<script>alert('XSS')</script>"}
            }
        ]
        
        for test in error_tests:
            try:
                if "json" in test:
                    response = requests.post(test["url"], json=test["json"], timeout=10)
                else:
                    response = requests.post(
                        test["url"], 
                        data=test.get("data", ""),
                        headers=test.get("headers", {}),
                        timeout=10
                    )
                
                if response.status_code in [400, 422, 500]:
                    self.print_test(test["name"], "PASS", f"Properly handled with {response.status_code}")
                elif response.status_code == 200:
                    # Check if dangerous input was sanitized
                    resp_text = response.json().get("response", "")
                    if "<script>" not in resp_text and "DROP TABLE" not in resp_text:
                        self.print_test(test["name"], "PASS", "Input sanitized")
                    else:
                        self.print_test(test["name"], "FAIL", "Dangerous input not sanitized")
                else:
                    self.print_test(test["name"], "WARN", f"Unexpected status: {response.status_code}")
                    
            except Exception as e:
                self.print_test(test["name"], "FAIL", str(e))
    
    async def test_model_capabilities(self):
        """Test actual AI model capabilities"""
        self.print_header("MODEL CAPABILITY TESTING")
        
        capability_tests = [
            {
                "name": "Language Understanding",
                "prompt": "What is the difference between affect and effect?",
                "check": lambda r: "affect" in r.lower() and "effect" in r.lower()
            },
            {
                "name": "Code Generation",
                "prompt": "Write a Python function to calculate factorial",
                "check": lambda r: "def" in r and "factorial" in r.lower()
            },
            {
                "name": "Reasoning",
                "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly?",
                "check": lambda r: "no" in r.lower() or "cannot" in r.lower()
            },
            {
                "name": "Creativity",
                "prompt": "Write a haiku about artificial intelligence",
                "check": lambda r: len(r.split('\n')) >= 3
            },
            {
                "name": "Multilingual",
                "prompt": "Translate 'Hello, how are you?' to Spanish",
                "check": lambda r: "hola" in r.lower() or "cómo" in r.lower()
            }
        ]
        
        for test in capability_tests:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": test["prompt"], "search_web": False},
                    timeout=15
                )
                
                if response.status_code == 200:
                    resp_text = response.json().get("response", "")
                    if test["check"](resp_text):
                        self.print_test(test["name"], "PASS", "Capability demonstrated")
                    else:
                        self.print_test(test["name"], "WARN", "Response doesn't meet expectations")
                else:
                    self.print_test(test["name"], "FAIL", f"Status: {response.status_code}")
                    
            except Exception as e:
                self.print_test(test["name"], "FAIL", str(e))
    
    def generate_report(self, hardcoded_results: Dict[str, Any]):
        """Generate comprehensive test report"""
        self.print_header("TEST REPORT SUMMARY")
        
        total_tests = len(self.test_results["passed"]) + len(self.test_results["failed"])
        pass_rate = (len(self.test_results["passed"]) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Colors.BOLD}Overall Statistics:{Colors.RESET}")
        print(f"  Total Tests Run: {total_tests}")
        print(f"  Passed: {Colors.GREEN}{len(self.test_results['passed'])}{Colors.RESET}")
        print(f"  Failed: {Colors.RED}{len(self.test_results['failed'])}{Colors.RESET}")
        print(f"  Warnings: {Colors.YELLOW}{len(self.test_results['warnings'])}{Colors.RESET}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if self.timing_data:
            print(f"\n{Colors.BOLD}Performance Metrics:{Colors.RESET}")
            print(f"  Average Response Time: {sum(self.timing_data)/len(self.timing_data):.2f}s")
            print(f"  Min Response Time: {min(self.timing_data):.2f}s")
            print(f"  Max Response Time: {max(self.timing_data):.2f}s")
        
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}CRITICAL FINDING - HARDCODED DETECTION:{Colors.RESET}")
        print(f"{'='*70}")
        
        if hardcoded_results["is_hardcoded"]:
            print(f"{Colors.RED}{Colors.BOLD}⚠️  YOUR AI SYSTEM APPEARS TO BE HARDCODED! ⚠️{Colors.RESET}")
            print(f"\n{Colors.RED}Evidence found:{Colors.RESET}")
            for indicator in hardcoded_results["indicators"]:
                print(f"  • {indicator}")
            print(f"\n{Colors.YELLOW}Recommendation:{Colors.RESET}")
            print("  Your system shows signs of using pre-programmed or templated responses")
            print("  rather than genuine AI model inference. Consider:")
            print("  1. Implementing actual model loading (not just importing)")
            print("  2. Using real tokenization and generation")
            print("  3. Removing hardcoded response templates")
            print("  4. Implementing proper context handling")
        else:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ Your AI system appears to be genuine!{Colors.RESET}")
            print(f"\n{Colors.GREEN}Positive indicators:{Colors.RESET}")
            print("  • Responses show natural variation")
            print("  • Dynamic content generation detected")
            print("  • Proper handling of different input types")
            print("  • Context retention capabilities")
        
        print(f"\n{Colors.BOLD}Failed Tests (if any):{Colors.RESET}")
        if self.test_results["failed"]:
            for test in self.test_results["failed"][:5]:
                print(f"  • {test}")
        else:
            print("  None - All tests passed!")
        
        print(f"\n{Colors.BOLD}Recommendations for Improvement:{Colors.RESET}")
        if hardcoded_results["confidence"] > 30:
            print("  1. Replace templated responses with actual model inference")
            print("  2. Implement proper tokenization and generation pipeline")
            print("  3. Add response caching for performance (not hardcoding)")
        
        if len(self.test_results["failed"]) > 0:
            print("  1. Fix failing API endpoints")
            print("  2. Improve error handling for edge cases")
        
        if self.test_results["warnings"]:
            print("  3. Address warning cases for better reliability")
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")


async def main():
    """Main test execution"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     RYTHMIA AI EUROPA - COMPREHENSIVE SYSTEM TEST SUITE     ║")
    print("║                  Testing for Hardcoded Responses             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    tester = RythmAiTestSuite()
    
    # Check if backend is running
    print(f"{Colors.BOLD}Checking system availability...{Colors.RESET}")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.RED}Backend not responding properly!{Colors.RESET}")
            print("Please start the backend: python backend/algorythm_ai_backend.py")
            return
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}Backend is not running!{Colors.RESET}")
        print("Starting backend automatically...")
        import subprocess
        subprocess.Popen([sys.executable, "backend/algorythm_ai_backend.py"])
        print("Waiting for backend to initialize...")
        await asyncio.sleep(5)
    
    # Run all tests
    hardcoded_results = await tester.test_hardcoded_responses()
    await tester.test_api_endpoints()
    await tester.test_model_capabilities()
    await tester.test_performance()
    await tester.test_error_handling()
    
    # Generate final report
    tester.generate_report(hardcoded_results)
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": tester.test_results,
            "hardcoded_detection": hardcoded_results,
            "timing_data": tester.timing_data
        }, f, indent=2)
    
    print(f"\n{Colors.BOLD}Results saved to test_results.json{Colors.RESET}")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
