#!/bin/bash

# Digital Twin Generator API Test Script
# Test all endpoints with curl commands

BASE_URL="http://localhost:5000"

echo "🎯 Digital Twin Generator - API Test Suite"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Test 1: Health Check
echo -e "${BLUE}1. Testing Health Check...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/health")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "Health Check (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 2: System Status
echo -e "${BLUE}2. Testing System Status...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/system-status")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "System Status (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 3: Quality Modes
echo -e "${BLUE}3. Testing Quality Modes...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/quality-modes")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "Quality Modes (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 4: Upload Selfies (with sample images)
echo -e "${BLUE}4. Testing Upload Selfies...${NC}"
echo "Note: This test requires sample images in the current directory"
echo "Creating sample test images..."

# Create a test directory with sample images
mkdir -p test_images
for i in {1..15}; do
    # Create a simple test image (1x1 pixel PNG)
    convert -size 512x512 xc:white test_images/selfie_$i.png 2>/dev/null || \
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" | base64 -d > test_images/selfie_$i.png
done

# Test upload with multiple files
if [ -d "test_images" ]; then
    response=$(curl -s -w "%{http_code}" \
        -F "files=@test_images/selfie_1.png" \
        -F "files=@test_images/selfie_2.png" \
        -F "files=@test_images/selfie_3.png" \
        -F "files=@test_images/selfie_4.png" \
        -F "files=@test_images/selfie_5.png" \
        -F "files=@test_images/selfie_6.png" \
        -F "files=@test_images/selfie_7.png" \
        -F "files=@test_images/selfie_8.png" \
        -F "files=@test_images/selfie_9.png" \
        -F "files=@test_images/selfie_10.png" \
        -F "files=@test_images/selfie_11.png" \
        -F "files=@test_images/selfie_12.png" \
        -F "files=@test_images/selfie_13.png" \
        -F "files=@test_images/selfie_14.png" \
        -F "files=@test_images/selfie_15.png" \
        "$BASE_URL/upload")
    
    http_code="${response: -3}"
    body="${response%???}"
    print_status $http_code "Upload Selfies (Status: $http_code)"
    echo "Response: $body"
    
    # Extract job_id from response if successful
    if [ $http_code -eq 200 ]; then
        JOB_ID=$(echo "$body" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
        echo "Job ID: $JOB_ID"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping upload test - no test images available${NC}"
fi
echo ""

# Test 5: Check Job Status (if we have a job_id)
if [ ! -z "$JOB_ID" ]; then
    echo -e "${BLUE}5. Testing Job Status...${NC}"
    response=$(curl -s -w "%{http_code}" "$BASE_URL/status/$JOB_ID")
    http_code="${response: -3}"
    body="${response%???}"
    print_status $http_code "Job Status (Status: $http_code)"
    echo "Response: $body"
    echo ""
    
    # Wait a bit and check again
    echo "Waiting 10 seconds and checking status again..."
    sleep 10
    response=$(curl -s -w "%{http_code}" "$BASE_URL/status/$JOB_ID")
    http_code="${response: -3}"
    body="${response%???}"
    print_status $http_code "Job Status (After 10s) (Status: $http_code)"
    echo "Response: $body"
    echo ""
else
    echo -e "${YELLOW}⚠️  Skipping job status test - no job ID available${NC}"
    echo ""
fi

# Test 6: Generate Avatar
echo -e "${BLUE}6. Testing Generate Avatar...${NC}"
response=$(curl -s -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "a realistic cinematic portrait of a woman in cyberpunk city background",
        "num_images": 1,
        "quality_mode": "high_fidelity"
    }' \
    "$BASE_URL/generate")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "Generate Avatar (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 7: List Jobs
echo -e "${BLUE}7. Testing List Jobs...${NC}"
response=$(curl -s -w "%{http_code}" "$BASE_URL/jobs")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "List Jobs (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 8: Cleanup
echo -e "${BLUE}8. Testing Cleanup...${NC}"
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/cleanup")
http_code="${response: -3}"
body="${response%???}"
print_status $http_code "Cleanup (Status: $http_code)"
echo "Response: $body"
echo ""

# Test 9: Download (if we have a job_id and it's completed)
if [ ! -z "$JOB_ID" ]; then
    echo -e "${BLUE}9. Testing Download...${NC}"
    response=$(curl -s -w "%{http_code}" "$BASE_URL/download/$JOB_ID/avatar_001.png")
    http_code="${response: -3}"
    print_status $http_code "Download (Status: $http_code)"
    if [ $http_code -eq 200 ]; then
        echo "✅ File downloaded successfully"
    else
        echo "❌ File not found or job not completed"
    fi
    echo ""
fi

# Cleanup test files
if [ -d "test_images" ]; then
    rm -rf test_images
    echo "🧹 Cleaned up test images"
fi

echo ""
echo -e "${GREEN}🎉 API Test Suite Completed!${NC}"
echo ""
echo "📋 Test Summary:"
echo "• Health Check: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• System Status: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• Quality Modes: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• Upload Selfies: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• Generate Avatar: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• List Jobs: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo "• Cleanup: $(if [ $http_code -eq 200 ]; then echo "✅"; else echo "❌"; fi)"
echo ""
echo "🚀 To test with real images:"
echo "curl -F 'files=@image1.jpg' -F 'files=@image2.jpg' ... http://localhost:5000/upload"
echo ""
echo "🎨 To generate an avatar:"
echo "curl -H 'Content-Type: application/json' -d '{\"prompt\":\"your prompt here\"}' http://localhost:5000/generate" 