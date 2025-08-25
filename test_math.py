import re

def calculate_math(expression: str) -> str:
    """Handle mathematical calculations"""
    print(f"Input: {expression}")
    try:
        # Check for math problems with question format
        if '?' in expression:
            # Extract just the math part
            math_part = expression.replace('?', '').strip()
            if 'what is' in math_part.lower():
                math_part = math_part.lower().replace('what is', '').strip()
            print(f"Math part: {math_part}")
        else:
            math_part = expression
        
        # Check for basic operations
        if '+' in math_part:
            numbers = re.findall(r'\d+', math_part)
            print(f"Found numbers for addition: {numbers}")
            if len(numbers) >= 2:
                nums = [int(n) for n in numbers]
                result = sum(nums)
                return str(result)
        elif '*' in math_part or 'x' in math_part.lower():
            numbers = re.findall(r'\d+', math_part)
            print(f"Found numbers for multiplication: {numbers}")
            if len(numbers) >= 2:
                nums = [int(n) for n in numbers]
                result = nums[0]
                for n in nums[1:]:
                    result *= n
                return str(result)
        elif '-' in math_part:
            numbers = re.findall(r'\d+', math_part)
            if len(numbers) >= 2:
                nums = [int(n) for n in numbers]
                return str(nums[0] - sum(nums[1:]))
        
        # Handle word problems
        numbers = re.findall(r'\d+', expression)
        if len(numbers) >= 2:
            if 'apples' in expression.lower() and ('give' in expression.lower() or 'away' in expression.lower()):
                nums = [int(n) for n in numbers]
                return str(nums[0] - nums[1])
        
        # Handle sequence
        if 'sequence' in expression.lower():
            numbers = re.findall(r'\d+', expression)
            if len(numbers) >= 3:
                # Check if it's doubling
                nums = [int(n) for n in numbers]
                if nums[1] == nums[0] * 2 and nums[2] == nums[1] * 2:
                    return str(nums[-1] * 2)
    except Exception as e:
        print(f"Error: {e}")
        pass
    return None

# Test cases
test_cases = [
    "What is 123 + 456?",
    "What is 50 * 3?",
    "If I have 17 apples and give away 9, how many do I have?",
    "What's the next number in the sequence: 2, 4, 8, 16, ?"
]

for test in test_cases:
    result = calculate_math(test)
    print(f"Test: {test}")
    print(f"Result: {result}")
    print("-" * 50)
