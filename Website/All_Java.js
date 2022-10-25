// Declaring constants
const cookieName="checked_rad2"
const expDays = 365
const classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

// POP UP FOR THE HELP BUTTON IN THE OPTIONS PAGE //
function HelpFunc() {
    let popup = document.getElementById("HPop");
    popup.classList.toggle("show")
}

// Function to set cookie
function setCookie(name, value, expires, path, domain, secure) {
    if (!expires){
        expires = new Date()
    }
    document.cookie = name + "=" + escape(value) +
        ((expires == null) ? "" : "; expires=" + expires.toUTCString()) +
        ((path == null) ? "" : "; path=" + path) +
        ((domain == null) ? "" : "; domain=" + domain) +
        ((secure == null) ? "" : "; secure")
}

// Function to retrieve a cookie
function getCookie(name) {
    let arg = name + "=";
    let alen = arg.length
    let clen = document.cookie.length
    let i = 0
    while (i < clen) {
        let j = i + alen
        if (document.cookie.substring(i, j) === arg) {
            return getCookieVal(j)
        }
        i = document.cookie.indexOf(" ", i) + 1
        if (i === 0) {
            break
        }
    }
    return null
}


//Function to get cookie value
function getCookieVal(offset) {
    let endstr = document.cookie.indexOf(";", offset)
    if (endstr === -1) {
        endstr = document.cookie.length
    }
    return unescape(document.cookie.substring(offset, endstr))
}

// Function that returns all the radio buttons' values
function getCheckedRad() {
    let radNum = getCookie(cookieName)

    if(!radNum) {
        return
    }

    let temp = radNum.split("&")

    for (let i = 0; i < temp.length; i++) {
        let el = document.getElementById("set" + i)
        let myInputs = el.getElementsByTagName("INPUT")
        myInputs[temp[i]].checked = true
    }
}

//Function that saves all the radio values
function saveCheckedRad() {
    let exp = new Date()
    exp.setTime(exp.getTime() + (expDays*365*24*60*60*1000))

    let c = 0
    let radNum = ""

    while (document.getElementById("set" + c)) {
        let el = document.getElementById("set" + c)
        let myInputs = el.getElementsByClassName("checkbox")

        for (let i = 0; i < myInputs.length; i++) {

            if (myInputs[i].checked) {
                radNum += i + "&"
            }
        }
        c++
    }
    setCookie(cookieName, radNum, exp)
}

//Set cookie specifically for Slider
function setSliderC(name, value, expire) {
    const d = new Date()
    d.setTime(d.getTime() + (expire * 24 * 60 * 60 * 1000))
    let expires = "expires=" + d.toUTCString()
    document.cookie = name + "=" + value + ";" + expires + ";path=/"
}

//get cookie specifically for Slider
function getSliderC(name) {
    let cname = name + "="
    let ca = document.cookie.split(";")
    for (let i = 0; i < ca.length; i ++) {
        let c = ca[i]
        while (c.charAt(0) === " ") {
            c = c.substring(1)
        }
        if (c.indexOf(cname) === 0) {
            return c.substring(cname.length, c.length)
        }
    }
    return null
}

//Function to save slider value on unload
function saveSliderVal() {
    let el = document.getElementById("Percentage")
    let cookie_name = "Percentage"
    setSliderC(cookie_name, el.value, 12)
}

//Function to return slider value onload
function getSliderVal() {
    let sliderC = getSliderC("Percentage")

    if (!sliderC) {return ""}

    let el = document.getElementById("Percentage")
    el.value = sliderC
}

//Main Popup Function
function openForm() {
    document.getElementById("popupForm").style.display = "block"
}

function closeForm() {
    document.getElementById("popupForm").style.display = "none"
}

function imgBox(){
    let img = document.getElementById("img")
    let blah = document.getElementById("blah")
    img.onchange = () => {
        const [file] = img.files
        if (file) {
            blah.src = URL.createObjectURL(file)
        }
    }
}

function fixImg() {
    let img = document.getElementById("blah")
    let box = document.getElementById("popupForm")
    let submit = document.getElementById("submit")
    if (img.height >= 32 && img.width >= 32) {
        submit.disabled = false

        if (img.height > box.clientHeight) {
            img.height = box.clientHeight - 420
        }
        if (img.width > box.clientWidth) {
            img.width = box.clientWidth - 5
        }
    }

    else {
        window.alert("The image must have dimensions greater than 32x32")
        submit.disabled = true
    }

    setTimeout(fixImg, 10000)
}

/*function submitImg() {
    let thing = document.getElementById('img')
    INPUT = thing.files
    console.log(INPUT)
}*/

/*function selectImg() {
    let ele = document.getElementById('finalImg')
    ele.src = URL.createObjectURL(INPUT)
}*/

/* all the stuff for big popup */
function openPop() {
    let input = document.getElementById("img")
    if(input.files[0]){
        document.getElementById("screen-pop").style.display = "block"
    }
    else{
        window.alert("you must select an image")
    }
}

function closePop() {
    document.getElementById("screen-pop").style.display = "none"
}

/* Traffic light Functions*/

function chooseColour(per) {
    if (per > 1){
        console.log("Error with Percentage Received")
    }
    else {
        if (per >= 0.75) {
            let ele = document.getElementById("green")
            ele.style.backgroundColor = "green"
        }
        else if (per < 0.75 && per >= 0.33) {
            let ele = document.getElementById("orng")
            ele.style.backgroundColor = "#FE8316"
        }
        else {
            let ele = document.getElementById("red")
            ele.style.backgroundColor = "red"
        }
    }
}

function getBig(array) {
    let max = Math.max.apply(Math, array)
    return array.indexOf(max)
}

/* Canvas function */
async function canvas() {
    let ele = document.getElementById("canvas")
    let ctx = ele.getContext("2d")
    let img = document.getElementById("blah")
    let newimg = document.getElementById("newimg")
    let trunc = 0.1
    let SliderC = getSliderC('Percentage')
    ctx.drawImage(img, 0, 0, 32, 32)
    newimg.src = ele.toDataURL("image/jpeg")
    newimg.height = 200
    newimg.width = 200

    let results = await predicter()
    let order = []
    let certainty = []
    for (let i = 0; i < 10; i++) {
        let val = getBig(results)
        order.push(val)
        certainty.push(results[val])
        results[val] = 0
    }

    if(SliderC) {trunc = SliderC/100}

    for (let i = 0; i < 10; i++) {
        let el = document.getElementById(i.toString())
        for (let j = 0; j < 10; j++) {
            if (order[i] === j) {
                if (certainty[i] < trunc) {}
                else {
                    el.innerText = classes[j] + " " + (certainty[i] * 100).toString().slice(0, 4) + " %"
                }
            }
        }
    }
    chooseColour(traffic())
    getAns()
}

async function loadModel() {
    return await tf.loadGraphModel('THE_JS_MODEL/model.json')
}

async function predicter() {
    let model =  await loadModel()
    let img = getData()
    let image = tf.browser.fromPixels(img).expandDims(0).toFloat()
    let y = await model.predict(image)
    return Array.from(y.dataSync())
}

function getData() {
    let el = document.getElementById('canvas')
    let ctx = el.getContext('2d');
    return ctx.getImageData(0, 0, el.width, el.height)
}

function traffic() {
    let el = document.getElementById('0').innerText
    let space = el.indexOf(" ")
    let part = el.substring(space + 1)
    let space2 = part.indexOf(" ")
    let final = part.substring(0, space2)
    return final/100
}

function clearList() {
    for (let i = 0; i < 10; i++) {
        let item = document.getElementById(i.toString())
        item.innerText = ""
    }
    let img = document.getElementById('newimg')
    img.src = ''
    let canvas = document.getElementById('canvas')
    let ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
}

tf.setBackend('cpu')


function getAns() {
    let ans = document.getElementById('0').innerText
    let box = document.getElementById('ans')
    let space = ans.indexOf(' ')
    ans = ans.substring(0, space)
    box.innerText += ' ' + ans
}